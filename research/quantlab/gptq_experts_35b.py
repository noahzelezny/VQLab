#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""E34: GPTQ error-compensated rounding for the 35B's 2-bit experts.

Streams the calibration corpus through the bf16 model layer-by-layer
(score_streaming's convention), captures each layer's SwitchGLU inputs +
routing (probe_routing_skew's spy pattern), builds PER-EXPERT Hessians from
the tokens actually routed there, and GPTQ-solves gate/up/down into MLX's
exact group-64 2-bit affine format. Results land as per-layer .npz
checkpoints (crash-safe, resumable); assemble_gptq_35b.py swaps them into
an RTN-built artifact.

v1 notes:
- fp-input variant: every layer sees FULL-PRECISION activations (we stream
  the bf16 model). Reference GPTQ propagates quantized outputs; if v1 shows
  a real win, that's the polish pass.
- grid = min/max affine per (row, 64-group) from the CURRENT compensated
  weights, snapped to bf16 (what the artifact stores) BEFORE rounding
  decisions, so decode uses exactly the grid the solver assumed.
"""
import gc
import json
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import mlx.core as mx

from gptq_solver import quantize_gptq, quantize_rtn, dequant

SRC = "/Volumes/Thunderbay SSD/Exo Models/Qwen--Qwen3.5-35B-A3B"
CKPT = pathlib.Path("/Volumes/Thunderbay SSD/Exo Models/rotlab-gptq-ckpt")
CORPUS = pathlib.Path(__file__).parent / "calib_corpus.txt"
SEQ = 4096
CHUNK = 4          # sequences per forward chunk
DAMP = 0.01

mx.set_cache_limit(8 << 30)


def bf16_snap(a):
    return np.array(mx.array(a.astype(np.float32)).astype(
        mx.bfloat16).astype(mx.float32))


def solve_one(args):
    """worker: GPTQ one expert. gu:[1024,2048] (gate;up stacked), H:[2048²],
    a is this expert's activations for the down Hessian."""
    e, Wgu, Hgu, Wd, Hd = args
    qi_gu, s_gu, b_gu = quantize_gptq(Wgu, Hgu, damp=DAMP)
    qi_d, s_d, b_d = quantize_gptq(Wd, Hd, damp=DAMP)
    return e, qi_gu, bf16_snap(s_gu), bf16_snap(b_gu), \
        qi_d, bf16_snap(s_d), bf16_snap(b_d)


def main():
    CKPT.mkdir(exist_ok=True)
    from mlx_lm.utils import load
    from mlx_lm.models import switch_layers
    with mx.stream(mx.cpu):
        model, tok, _ = load(SRC, lazy=True, return_config=True)
    core = model
    for name in ("language_model", "model"):
        while hasattr(core, name):
            core = getattr(core, name)
    n_layers = len(core.layers)

    text = CORPUS.read_text(errors="replace")
    ids = tok.encode(text)
    n_seq = len(ids) // SEQ
    toks = mx.array(ids[: n_seq * SEQ]).reshape(n_seq, SEQ)
    print(f"calib: {n_seq} seqs x {SEQ} tok, {n_layers} layers", flush=True)

    # spy on SwitchGLU — one layer is live at a time, so a global capture
    cap = {"x": [], "inds": []}
    orig_call = switch_layers.SwitchGLU.__call__

    def spy(self, x, indices, *a, **k):
        cap["x"].append(np.array(x.reshape(-1, x.shape[-1]).astype(mx.float32)))
        cap["inds"].append(np.array(indices.reshape(-1, indices.shape[-1])))
        return orig_call(self, x, indices, *a, **k)

    with mx.stream(mx.cpu):
        mx.eval(core.embed_tokens.parameters())
    hs = []
    for s0 in range(0, n_seq, CHUNK):
        h = core.embed_tokens(toks[s0:s0 + CHUNK])
        mx.eval(h)
        hs.append(h)

    pool = ProcessPoolExecutor(max_workers=6)
    for li in range(n_layers):
        out_f = CKPT / f"layer{li:02d}.npz"
        blk = core.layers[li]
        mask = None if blk.is_linear else "causal"
        with mx.stream(mx.cpu):
            mx.eval(blk.parameters())
        blk.eval()

        t0 = time.time()
        switch_layers.SwitchGLU.__call__ = spy
        cap["x"], cap["inds"] = [], []
        new_hs = []
        for h in hs:
            ho = blk(h, mask=mask, cache=None)
            mx.eval(ho)
            new_hs.append(ho)
        switch_layers.SwitchGLU.__call__ = orig_call
        hs = new_hs

        if out_f.exists():   # resumable: forward still had to advance hs
            print(f"layer {li:2d} ckpt exists, skipped solve", flush=True)
            core.layers[li] = None
            del blk
            gc.collect()
            mx.clear_cache()
            continue

        X = np.concatenate(cap["x"])          # [T, 2048]
        I = np.concatenate(cap["inds"])       # [T, 8]
        cap["x"], cap["inds"] = [], []

        # bf16 weights for this layer's experts
        sm = blk.mlp.switch_mlp
        Wg = np.array(sm.gate_proj.weight.astype(mx.float32))  # [256,512,2048]
        Wu = np.array(sm.up_proj.weight.astype(mx.float32))
        Wd = np.array(sm.down_proj.weight.astype(mx.float32))  # [256,2048,512]

        jobs = []
        for e in range(Wg.shape[0]):
            rows = X[(I == e).any(axis=1)]
            if rows.shape[0] < 64:   # starved expert: fall back to RTN later
                jobs.append((e, None, None, None, None))
                continue
            Hgu = rows.T @ rows
            g = rows @ Wg[e].T
            u = rows @ Wu[e].T
            a = (g * (1.0 / (1.0 + np.exp(-g)))) * u   # silu(g)*u
            Hd = a.T @ a
            Wgu = np.concatenate([Wg[e], Wu[e]], axis=0)  # [1024,2048]
            jobs.append((e, Wgu, Hgu, Wd[e], Hd))

        res = {}
        solve_jobs = [j for j in jobs if j[1] is not None]
        for r in pool.map(solve_one, solve_jobs, chunksize=4):
            res[r[0]] = r[1:]
        starved = [j[0] for j in jobs if j[1] is None]
        for e in starved:
            qg, sg, bg = quantize_rtn(np.concatenate([Wg[e], Wu[e]], axis=0))
            qd, sd, bd = quantize_rtn(Wd[e])
            res[e] = (qg, bf16_snap(sg), bf16_snap(bg),
                      qd, bf16_snap(sd), bf16_snap(bd))

        E = Wg.shape[0]
        np.savez_compressed(
            out_f,
            qi_gu=np.stack([res[e][0] for e in range(E)]),
            s_gu=np.stack([res[e][1] for e in range(E)]),
            b_gu=np.stack([res[e][2] for e in range(E)]),
            qi_d=np.stack([res[e][3] for e in range(E)]),
            s_d=np.stack([res[e][4] for e in range(E)]),
            b_d=np.stack([res[e][5] for e in range(E)]),
            starved=np.array(starved),
        )
        print(f"layer {li:2d} solved ({len(starved)} starved/RTN) "
              f"in {time.time()-t0:.0f}s", flush=True)
        core.layers[li] = None
        del blk, X, I, Wg, Wu, Wd
        gc.collect()
        mx.clear_cache()
    pool.shutdown()
    print("ALL LAYERS DONE", flush=True)


if __name__ == "__main__":
    main()
