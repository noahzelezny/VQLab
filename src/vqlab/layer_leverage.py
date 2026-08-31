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

Known issues, FIXED 2026-08-29 (verified equal-output on a tiny resident
pair; large-model peak is unmeasured until the next real probe run — per
the probe-duration rule, treat the improvement as expected, not measured):
  - the three per-layer forwards were eval'd as one batch with both models'
    layer weights resident (112.3 GiB peak on the 96 GiB M3 vs the 335 GiB
    teacher) — now evaluated sequentially, teacher layer freed first;
  - models were loaded outside a CPU-stream block, so the lazy shard-read
    ops were GPU-stream-bound at creation (the IV.1 SMB watchdog class,
    seen on the M4 ~22:40 08-29) — loads now happen under mx.stream(mx.cpu).
"""
import argparse
import gc
import importlib
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
        # MEMORY (fix 2026-08-29, see docstring known-issues): the original
        # built all three forwards and eval'd them as ONE batch — one command
        # buffer holding three layer graphs' intermediates, with BOTH models'
        # layer weights resident throughout (112.3 GiB peak on the 96 GiB M3
        # against the 335 GiB teacher). Now each forward is eval'd alone, and
        # the teacher's layer weights are released before the student runs.
        # Same ops, same values — only concurrency changes.
        t_out = t_blk(h_t, t_core.rope, t_mask, t_conv, None, None,
                      ids, t_prev)
        mx.eval(t_out)
        t_core.layers[i] = None                 # teacher weights out of the
        del t_blk                               # peak before student runs
        s_local = s_blk(h_t, s_core.rope, s_mask, s_conv, None, None,
                        ids, s_prev)                    # injected: teacher h
        mx.eval(s_local)
        local = _rel(s_local, t_out)
        del s_local                             # freed before the 3rd graph
        s_traj = s_blk(h_s, s_core.rope, s_mask, s_conv, None, None,
                       ids, s_prev)                     # student trajectory
        mx.eval(s_traj)
        traj = _rel(s_traj, t_out)
        rows.append({"layer": i, "local_rel": round(local, 6),
                     "traj_rel": round(traj, 6)})
        # rebind trajectories, dropping the PREVIOUS layer's h_t/h_s before
        # collect so no stale activation survives into the next iteration.
        h_t, h_s = t_out, s_traj
        s_core.layers[i] = None
        del t_out, s_traj, s_blk
        gc.collect()
        mx.clear_cache()
        print(f"  layer {i}/{n-1}  local {local:.4f}  traj {traj:.4f}  "
              f"{time.time()-t0:.1f}s "
              f"(peak {mx.get_peak_memory()/1024**3:.1f}G)", flush=True)
    return rows


def probe_glm5_next(teacher, student, ids_list, args):
    """Mirrors stream_score.score_glm5_next (rule-5 validated, bitwise,
    2026-08-29): masks on pre-broadcast h, hc_mult broadcast + contiguous,
    per-layer is_linear mask dispatch, layer(x, mask, cache). The hc mean /
    final norm / head are irrelevant here — the probe compares hidden
    states, so trajectories stay in the (B, S, hc, D) representation.
    Memory pattern identical to probe_qwen4_exp's fixed loop."""
    import importlib

    rows = []
    ids = mx.array([ids_list[:-1]])

    def prologue(model):
        lm = getattr(model, "language_model", model)
        core = lm.model
        with mx.stream(mx.cpu):
            mx.eval(core.embed_tokens.parameters())
        h = core.embed_tokens(ids)
        lang = importlib.import_module(type(core).__module__)
        attn_mask = lang.create_attention_mask(h, None, return_array=True)
        ssm_mask = lang.create_ssm_mask(h, None)
        h = mx.broadcast_to(h[:, :, None, :],
                            (*h.shape[:2], core.hc_mult, h.shape[-1]))
        h = mx.contiguous(h)
        mx.eval(h)
        return core, h, attn_mask, ssm_mask

    t_core, h_t, t_attn, t_ssm = prologue(teacher)
    s_core, h_s, s_attn, s_ssm = prologue(student)
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
        t_mask = t_ssm if t_blk.is_linear else t_attn
        s_mask = s_ssm if s_blk.is_linear else s_attn
        t_out = t_blk(h_t, mask=t_mask, cache=None)
        mx.eval(t_out)
        t_core.layers[i] = None
        del t_blk
        s_local = s_blk(h_t, mask=s_mask, cache=None)
        mx.eval(s_local)
        local = _rel(s_local, t_out)
        del s_local
        s_traj = s_blk(h_s, mask=s_mask, cache=None)
        mx.eval(s_traj)
        traj = _rel(s_traj, t_out)
        rows.append({"layer": i, "local_rel": round(local, 6),
                     "traj_rel": round(traj, 6)})
        h_t, h_s = t_out, s_traj
        s_core.layers[i] = None
        del t_out, s_traj, s_blk
        gc.collect()
        mx.clear_cache()
        print(f"  layer {i}/{n-1}  local {local:.4f}  traj {traj:.4f}  "
              f"{time.time()-t0:.1f}s "
              f"(peak {mx.get_peak_memory()/1024**3:.1f}G)", flush=True)
    return rows


PROBES = {
    "qwen4_exp": {"fn": probe_qwen4_exp, "family": "qwen4_exp"},
    # glm5_next: loop validated on a tiny resident pair (see docstring);
    # runs under the mlx_vlm runtime via runtime_load, glm5vlm venv.
    "glm5_next": {"fn": probe_glm5_next, "family": "glm5_next"},
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
    # Cap MLX's buffer cache: with three forwards per layer the cache can
    # hold a layer's worth of freed intermediates between clear_cache calls.
    mx.set_cache_limit(8 << 30)
    # IMPORT THE ARCH FIRST, OUTSIDE ANY CPU-STREAM BLOCK (fix 2026-08-30).
    # mlx-vlm's hyper_connection builds its fused sinkhorn Metal kernel AT
    # IMPORT TIME behind `if mx.default_device() != mx.gpu: return None`, and
    # a cpu-stream context reports default_device() as CPU. Importing the arch
    # inside the watchdog block below therefore leaves that kernel permanently
    # None, and the first GPU forward dies with "'NoneType' object is not
    # callable" -- 40 minutes into a streamed pass. The two concerns are
    # separable: the watchdog needs the WEIGHT-READ ops created under the CPU
    # stream, not the module import.
    importlib.import_module(
        f"{runtime_load.runtime_for(entry['family'])}.models.{mt}")
    # WATCHDOG (fix 2026-08-29): a stream binds at OP-CREATION time, and the
    # lazy shard-read ops are created HERE, inside load_model — not at the
    # per-layer eval. Loading outside a CPU-stream block leaves every weight
    # read on the GPU stream, where an SMB stall inside a command buffer is
    # a Metal watchdog kill (IV.1's class; bit fit_ple the same way). The
    # per-layer `with mx.stream(mx.cpu): mx.eval(...)` was never enough on
    # its own — creation is what binds.
    with mx.stream(mx.cpu):
        teacher, _ = runtime_load.load_for_family(entry["family"],
                                                  pathlib.Path(a.teacher),
                                                  lazy=True)
        student, _ = runtime_load.load_for_family(entry["family"], sp,
                                                  lazy=True)
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
