#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""E34c probe: is the per-expert Hessian converged at 3.1x oversampling?

Decides whether MORE CALIBRATION DATA is a real lever before spending 2h on
it. Splits the calibration tokens into two disjoint halves, builds H_A / H_B
independently for the same experts, and asks the only question that matters:
do they produce the SAME GPTQ rounding decisions?

Interpretation:
- solutions agree  -> Hessian converged; more data cannot help; the n=4
  activation-fitted finding stands and we go to vector quant.
- solutions differ -> compensation is being fit to sampling noise; more data
  is the lever; the 8x run is justified.

Reference scale: the two halves are each HALF of current data, so any
disagreement here is an UPPER bound on what full-data-vs-2x would show.
Also reports the RTN control: how much do the two halves disagree about a
decision that uses no Hessian at all (must be ~0 — a sanity floor).
"""
import gc
import numpy as np
import mlx.core as mx

from gptq_solver import quantize_gptq, quantize_rtn, dequant

mx.set_cache_limit(8 << 30)

SRC = "/Volumes/Thunderbay SSD/Exo Models/Qwen--Qwen3.5-35B-A3B"
ACT = "/Volumes/Thunderbay SSD/Exo Models/rotlab-35B-base-struct6"
LAYERS = (0, 12, 24, 36)
N_EXPERTS = 12          # sampled per layer
SEQ = 4096
CHUNK = 4


def main():
    from mlx_lm.utils import load
    from mlx_lm.models import switch_layers
    with mx.stream(mx.cpu):
        model, tok, _ = load(ACT, lazy=True, return_config=True)
        wmodel, _, _ = load(SRC, lazy=True, return_config=True)

    def _core(m):
        c = m
        for n in ("language_model", "model"):
            while hasattr(c, n):
                c = getattr(c, n)
        return c
    core, wcore = _core(model), _core(wmodel)

    text = open("calib_corpus.txt", errors="replace").read()
    ids = tok.encode(text)
    n_seq = len(ids) // SEQ
    toks = mx.array(ids[: n_seq * SEQ]).reshape(n_seq, SEQ)
    print(f"calib {n_seq} seqs; halves of {n_seq//2} seqs each", flush=True)

    cap = {"x": [], "i": []}
    orig = switch_layers.SwitchGLU.__call__

    def spy(self, x, indices, *a, **k):
        cap["x"].append(np.array(x.reshape(-1, x.shape[-1]).astype(mx.float32)))
        cap["i"].append(np.array(indices.reshape(-1, indices.shape[-1])))
        return orig(self, x, indices, *a, **k)

    with mx.stream(mx.cpu):
        mx.eval(core.embed_tokens.parameters())
    hs = [core.embed_tokens(toks[s:s + CHUNK]) for s in range(0, n_seq, CHUNK)]
    mx.eval(hs)
    # which captured rows belong to the first half of sequences
    half_seq = n_seq // 2

    rows_agree, rows_rtn = [], []
    for li in range(max(LAYERS) + 1):
        blk = core.layers[li]
        mask = None if blk.is_linear else "causal"
        with mx.stream(mx.cpu):
            mx.eval(blk.parameters())
        blk.eval()
        want = li in LAYERS
        if want:
            switch_layers.SwitchGLU.__call__ = spy
            cap["x"], cap["i"] = [], []
        new = []
        for h in hs:
            o = blk(h, mask=mask, cache=None)
            mx.eval(o)
            new.append(o)
        hs = new
        if want:
            switch_layers.SwitchGLU.__call__ = orig
            # rows are emitted chunk by chunk; first half of CHUNKS = first half
            n_chunks = len(cap["x"])
            cut = 0
            for c in range(n_chunks):
                if c * CHUNK < half_seq:
                    cut += cap["x"][c].shape[0]
            X = np.concatenate(cap["x"])
            I = np.concatenate(cap["i"])
            cap["x"], cap["i"] = [], []
            XA, IA = X[:cut], I[:cut]
            XB, IB = X[cut:], I[cut:]

            wblk = wcore.layers[li]
            with mx.stream(mx.cpu):
                mx.eval(wblk.parameters())
            sm = wblk.mlp.switch_mlp
            Wg = np.array(sm.gate_proj.weight.astype(mx.float32))
            Wu = np.array(sm.up_proj.weight.astype(mx.float32))

            step = max(1, 256 // N_EXPERTS)
            agree, rtnagree, ns = [], [], []
            for e in range(0, 256, step):
                ra = XA[(IA == e).any(axis=1)]
                rb = XB[(IB == e).any(axis=1)]
                if min(ra.shape[0], rb.shape[0]) < 256:
                    continue
                W = np.concatenate([Wg[e], Wu[e]], axis=0)
                qa, _, _ = quantize_gptq(W, ra.T @ ra)
                qb, _, _ = quantize_gptq(W, rb.T @ rb)
                agree.append(float((qa == qb).mean()))
                qr, _, _ = quantize_rtn(W)
                rtnagree.append(float((qa == qr).mean()))
                ns.append(min(ra.shape[0], rb.shape[0]))
            rows_agree += agree
            rows_rtn += rtnagree
            print(f"layer {li:2d}: {len(agree)} experts, ~{np.mean(ns):.0f} tok/half"
                  f"  GPTQ_A vs GPTQ_B agree {np.mean(agree)*100:5.1f}%"
                  f"   (GPTQ vs RTN agree {np.mean(rtnagree)*100:5.1f}%)",
                  flush=True)
            wcore.layers[li] = None
        core.layers[li] = None
        del blk
        gc.collect()
        mx.clear_cache()

    a, r = np.mean(rows_agree), np.mean(rows_rtn)
    print(f"\nOVERALL  GPTQ(half A) vs GPTQ(half B): {a*100:.1f}% identical")
    print(f"         GPTQ vs RTN                 : {r*100:.1f}% identical")
    print(f"\nGPTQ moves {(1-r)*100:.1f}% of weights off the RTN choice;")
    print(f"of those decisions, the two halves DISAGREE on "
          f"{(1-a)*100:.1f}% of ALL weights.")
    print("VERDICT:", "NOISE-DOMINATED — more data is the lever"
          if (1 - a) > 0.4 * (1 - r) else
          "CONVERGED — more data will not help")


if __name__ == "__main__":
    main()
