"""Layer-leverage probe: which layers' quantization damage matters most.

One interleaved streamed pass over teacher + student (stream_score's E18
layer-streaming, both models lazy, one layer of each resident at a time).
At every layer both blocks are fed the TEACHER's hidden state, giving
per-layer LOCAL damage with no compounding; the student's own trajectory
is advanced alongside, giving the compounding curve for free.

Per layer, two numbers:
  local_rel  ||S_i(h_t) - T_i(h_t)|| / ||T_i(h_t)||   (damage injected here)
  traj_rel   ||h_s - h_t|| / ||h_t|| after layer i    (accumulated drift;
             a JUMP between consecutive layers marks a high-leverage layer)

This is a RANKING instrument for allocation decisions (which layers earn
bigger K), not a quality score: hidden-state norms are not output KL, and
the verdict on any mixed build is still the referee + KL scorer.

FAMILY SUPPORT IS EXPLICIT, same argument as stream_score: a streamed loop
re-implements the forward; unknown model_type is a hard error.

    python -m vqlab.cli layer-leverage --teacher <bf16 dir> --student <dir>
        --corpus <txt> [--tokens N] [--out probe.json]
"""
import argparse
import gc
import json
import pathlib
import sys
import time

import mlx.core as mx

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import runtime_load


def _rel(a, b):
    """||a - b|| / ||b||, computed in float32 off the fp16 activations."""
    a32 = a.astype(mx.float32)
    b32 = b.astype(mx.float32)
    num = mx.sqrt(mx.sum(mx.square(a32 - b32)))
    den = mx.sqrt(mx.sum(mx.square(b32)))
    return float((num / den).item())


def probe_qwen4_exp(teacher, student, ids_list, args):
    """Mirrors stream_score.score_qwen4_exp's forward exactly (same layer
    signature, same hc tiling, ids/prev_ctx fed so the PLE path runs)."""
    from mlx_lm.models.qwen4_exp import create_attention_mask, create_ssm_mask

    rows = []
    ids = mx.array([ids_list[:-1]])

    def prologue(model):
        core = model.model
        with mx.stream(mx.cpu):
            mx.eval(core.embed_tokens.parameters())
        h = core.embed_tokens(ids)
        mask = create_attention_mask(h, None)
        lin = [i for i, l in enumerate(core.layers)
               if l.layer_type == "linear_attention"]
        conv_mask = create_ssm_mask(h, None) if lin else None
        prev_ctx = None
        if core.ple_layers:
            ctx = core.args.ngram_size - 1
            eos = core.args.eos_token_id
            eos = eos[0] if isinstance(eos, list) else eos
            prev_ctx = mx.full((ids.shape[0], ctx), eos, ids.dtype)
        h = mx.tile(h, (1, 1, core.hc))
        mx.eval(h)
        return core, h, mask, conv_mask, prev_ctx

    t_core, h_t, t_mask, t_conv, t_prev = prologue(teacher)
    s_core, h_s, s_mask, s_conv, s_prev = prologue(student)
    # embeddings are not VQ'd, so the two trajectories must start identical;
    # a mismatch here means the models disagree before any expert runs.
    start_rel = _rel(h_s, h_t)
    if start_rel > 1e-3:
        print(f"WARNING: trajectories differ at the embedding "
              f"(rel {start_rel:.2e}) — local numbers are still valid, "
              f"traj numbers include this offset.", flush=True)

    n = len(t_core.layers)
    if len(s_core.layers) != n:
        raise SystemExit(f"FAIL: layer count mismatch (teacher {n}, "
                         f"student {len(s_core.layers)})")
    for i in range(n):
        t_blk, s_blk = t_core.layers[i], s_core.layers[i]
        with mx.stream(mx.cpu):
            mx.eval(t_blk.parameters())
            mx.eval(s_blk.parameters())
        t0 = time.time()
        t_out = t_blk(h_t, t_core.rope, t_mask, t_conv, None, None,
                      ids, t_prev)
        s_local = s_blk(h_t, s_core.rope, s_mask, s_conv, None, None,
                        ids, s_prev)                    # injected: teacher h
        s_traj = s_blk(h_s, s_core.rope, s_mask, s_conv, None, None,
                       ids, s_prev)                     # student trajectory
        mx.eval(t_out, s_local, s_traj)
        local = _rel(s_local, t_out)
        traj = _rel(s_traj, t_out)
        rows.append({"layer": i, "local_rel": round(local, 6),
                     "traj_rel": round(traj, 6)})
        h_t, h_s = t_out, s_traj
        t_core.layers[i] = None
        s_core.layers[i] = None
        del t_blk, s_blk, t_out, s_local, s_traj
        gc.collect()
        mx.clear_cache()
        print(f"  layer {i}/{n-1}  local {local:.4f}  traj {traj:.4f}  "
              f"{time.time()-t0:.1f}s "
              f"(peak {mx.get_peak_memory()/1024**3:.1f}G)", flush=True)
    return rows


PROBES = {
    "qwen4_exp": {"fn": probe_qwen4_exp, "family": "qwen4_exp"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, help="bf16 teacher dir")
    ap.add_argument("--student", required=True, help="quantized artifact dir")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--tokens", type=int, default=1024,
                    help="ranking needs fewer tokens than a ladder score")
    ap.add_argument("--out", default=None, help="write per-layer JSON here")
    a = ap.parse_args()

    from mlx_lm.utils import load_tokenizer
    sp = pathlib.Path(a.student)
    cfg = json.load(open(sp / "config.json"))
    mt = cfg.get("model_type") or cfg.get("text_config", {}).get("model_type")
    if mt not in PROBES:
        raise SystemExit(f"FAIL: no layer-leverage probe for "
                         f"model_type={mt!r}. Supported: {sorted(PROBES)}.")
    entry = PROBES[mt]
    teacher, _ = runtime_load.load_for_family(entry["family"],
                                              pathlib.Path(a.teacher),
                                              lazy=True)
    student, _ = runtime_load.load_for_family(entry["family"], sp, lazy=True)
    print(runtime_load.resolved_runtime_note(student), flush=True)
    tok = load_tokenizer(sp)
    ids = tok.encode(open(a.corpus).read())[: a.tokens + 1]
    bos = getattr(tok, "bos_token_id", None)
    if bos is not None and (not ids or ids[0] != bos):
        ids = [bos] + ids[: a.tokens]

    rows = entry["fn"](teacher, student, ids, a)
    ranked = sorted(rows, key=lambda r: -r["local_rel"])
    rec = {"teacher": a.teacher, "student": str(sp), "corpus": a.corpus,
           "tokens": len(ids) - 1, "layers": rows,
           "top_local": [r["layer"] for r in ranked[:8]]}
    print(json.dumps({"top_local": rec["top_local"],
                      "final_traj_rel": rows[-1]["traj_rel"]}), flush=True)
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(rec, indent=1))
        print(f"per-layer record -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
