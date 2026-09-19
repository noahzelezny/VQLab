"""vqlab decode-timeline — every stage of ONE decode token, in order, timed.

WHY THIS EXISTS. The lab has deletion arms (`decode-ladder`) and derived
shares by subtraction (F136: "VQ linears ~24.6 ms = 54.5%"), but it has
never had the plain thing: a sequential list of every stage of a decode
step with a time against each, summing to the whole, sorted so the biggest
pieces are obvious. Deletion arms give UPPER BOUNDS that do not sum; a
timeline gives a PARTITION that does. You cannot rank what you have not
partitioned.

THE TRAP THIS INSTRUMENT IS BUILT AROUND (F133). The obvious implementation
-- wrap each module, `mx.eval` after each, record the delta -- measures
per-eval ROUND-TRIP LATENCY, not the op. F133 did exactly that and reported
parts summing to 1667 us against a 410 us whole, with a projection to four
outputs slower than the entire chain. Every number it produced was void.

So this measures CUMULATIVE PREFIXES instead: run the forward truncated
after stage k, eval once, record the total. Stage k's cost is
`cum[k] - cum[k-1]`. Each measurement is one real evaluation of a real
prefix, so there is no per-op round trip to absorb, and the deltas are
additive BY CONSTRUCTION.

AND IT CHECKS THAT. The sum of the deltas is compared against an
independently measured full step. If they disagree by more than
--additivity-tol, the run REFUSES to report a ranking, because a partition
that does not sum to the whole is not a partition. That check is the entire
difference between this and F133.

COST is O(n^2) work -- prefix k re-runs stages 0..k -- which is fine
because a decode step is ~40 ms and there are ~130 prefixes.

    vqlab decode-timeline --art <dir> [--context 64] [--reps 3]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

import mlx.core as mx

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from vqlab import runtime_load  # noqa: E402


def gpu_watts() -> float:
    """Instantaneous GPU package watts, or -1. Utilization is NOT usable:
    WindowServer pegs gpu_usage to 60-95% while drawing 2-4 W."""
    try:
        out = subprocess.run(
            ["/opt/homebrew/bin/macmon", "pipe", "--interval", "700", "-s", "1"],
            capture_output=True, text=True, timeout=6).stdout.splitlines()
        return float(json.loads(out[0])["gpu_power"])
    except Exception:
        return -1.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--art", required=True)
    ap.add_argument("--family", default="qwen3_5")
    ap.add_argument("--context", type=int, default=64,
                    help="prompt tokens prefilled before the timed decode "
                         "step. Full-attention layers cost more at longer "
                         "context; linear (GatedDeltaNet) layers do not, so "
                         "this changes the MIX and is stated in the output.")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-watts", type=float, default=12.0)
    ap.add_argument("--additivity-tol", type=float, default=5.0,
                    help="percent. If the deltas do not sum to an "
                         "independently measured full step within this, the "
                         "ranking is REFUSED (F133).")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()

    w0 = gpu_watts()
    if w0 > a.max_watts:
        raise SystemExit(
            f"REFUSING: GPU at {w0:.1f} W (limit {a.max_watts:.0f}). Rule III: "
            "never measure on a contended box.")

    model, _cfg = runtime_load.load_for_family(a.family, a.art, lazy=True)
    from mlx_lm.utils import load_tokenizer
    from mlx_lm.models.cache import make_prompt_cache
    tok = load_tokenizer(pathlib.Path(a.art))

    text = ("the quick brown fox jumps over the lazy dog while considering "
            "distributed inference ")
    ids = tok.encode(text * (a.context // 13 + 2))[: a.context]

    # Resolve the trunk by what it CARRIES, not by a guessed path. This
    # family nests model.language_model.model; requiring BOTH layers and
    # embed_tokens stops the walk at the right level (a wrapper can have
    # `layers` without `embed_tokens`, which is how the first version of
    # this landed on the wrong module).
    def _trunk(m, depth=0):
        if depth > 6:
            return None
        try:
            if hasattr(m, "layers") and hasattr(m, "embed_tokens"):
                return m
        except Exception:
            pass
        for attr in ("language_model", "model"):
            sub = getattr(m, attr, None)
            if sub is not None and sub is not m:
                got = _trunk(sub, depth + 1)
                if got is not None:
                    return got
        return None

    inner = _trunk(model)
    if inner is None:
        raise SystemExit("FAIL: could not locate a trunk carrying both "
                         "`layers` and `embed_tokens`")
    # the head lives on the module that OWNS the trunk
    head_owner = getattr(model, "language_model", model)
    layers = inner.layers
    arch = sys.modules[type(layers[0]).__module__]

    # Stage list, in execution order. Sub-layer granularity: the attention
    # half and the MLP half are separated because they are different
    # kernels with different levers (F131 found the regime inverts between
    # them), and lumping them would hide exactly what this is for.
    stages = [("embed", None, None)]
    for i, lyr in enumerate(layers):
        kind = "gdn" if getattr(lyr, "is_linear", False) else "attn"
        stages.append((f"L{i:02d}.{kind}", i, "attn"))
        stages.append((f"L{i:02d}.mlp", i, "mlp"))
    stages.append(("final_norm", None, None))
    stages.append(("lm_head", None, None))

    def fresh_cache():
        c = make_prompt_cache(model)
        model(mx.array([ids]), cache=c)      # untimed prefill, real work
        mx.eval([x for x in (c or []) if x is not None] or mx.array(0.0))
        return c

    def forward_prefix(k, cache, tokid):
        """Run the decode step truncated AFTER stage k, return the tensor."""
        h = inner.embed_tokens(tokid)
        if k == 0:
            return h
        fa = arch.create_attention_mask(h, cache[inner.fa_idx])
        ssm = arch.create_ssm_mask(h, cache[inner.ssm_idx])
        s = 1
        for i, lyr in enumerate(layers):
            mask = ssm if lyr.is_linear else fa
            att = lyr.linear_attn if lyr.is_linear else lyr.self_attn
            r = att(lyr.input_layernorm(h), mask, cache[i])
            h = h + r
            if s == k:
                return h
            s += 1
            h = h + lyr.mlp(lyr.post_attention_layernorm(h))
            if s == k:
                return h
            s += 1
        h = inner.norm(h)
        if s == k:
            return h
        tie = getattr(getattr(head_owner, "args", None),
                      "tie_word_embeddings", False)
        return (inner.embed_tokens.as_linear(h) if tie
                else head_owner.lm_head(h))

    tokid = mx.array([[ids[-1]]])

    def timed(k):
        best = float("inf")
        for _ in range(a.reps + 1):
            c = fresh_cache()
            mx.synchronize()
            t0 = time.time()
            out = forward_prefix(k, c, tokid)
            mx.eval(out)
            mx.synchronize()
            best = min(best, time.time() - t0)
            del c
        return best * 1e3

    print(f"\ndecode-timeline  {pathlib.Path(a.art).name}")
    print(f"context {a.context} tokens, best-of-{a.reps}, {len(stages)} stages"
          f"   GPU {w0:.1f} W\n", flush=True)

    cum, prev = [], 0.0
    rows = []
    for k, (name, _li, kind) in enumerate(stages):
        t = timed(k)
        cum.append(t)
        rows.append((name, kind, t - prev))
        prev = t
        if k % 16 == 0:
            print(f"  [{k:3d}/{len(stages)}] {name:16s} cum {t:8.2f} ms",
                  flush=True)

    whole = timed(len(stages) - 1)
    total = sum(d for _, _, d in rows)
    drift = abs(total - whole) / whole * 100

    w1 = gpu_watts()
    if w1 > a.max_watts:
        raise SystemExit(
            f"\nVOID: GPU ended at {w1:.1f} W (started {w0:.1f}). A foreign "
            "job landed mid-run; discard every number above.")

    print(f"\n  full step (independent)  {whole:8.2f} ms")
    print(f"  sum of stage deltas      {total:8.2f} ms   drift {drift:.2f}%")
    if drift > a.additivity_tol:
        raise SystemExit(
            f"\nREFUSING TO RANK: the deltas do not sum to the whole within "
            f"{a.additivity_tol}%. A partition that does not add up is not a "
            "partition, and ranking it would repeat F133. Investigate before "
            "quoting any stage.")
    print(f"  ADDITIVITY OK (<= {a.additivity_tol}%)")

    # NEGATIVE-DELTA GUARD, and it is STRICTLY STRONGER than additivity.
    # cum[k] < cum[k-1] means a longer prefix measured FASTER than a shorter
    # one, which cannot happen; it means run-to-run noise exceeds the stage
    # being measured. Additivity does NOT catch this -- a contended trial run
    # passed at 4.51% drift while reporting lm_head at -59 ms -- because
    # positive and negative noise cancel in the sum. The size of the most
    # negative delta is the honest NOISE FLOOR: no stage smaller than that
    # has been measured, only guessed.
    negs = [(n, d) for n, _k, d in rows if d < 0]
    floor = -min((d for _n, _k, d in rows), default=0.0)
    print(f"  noise floor (worst negative delta) {floor:8.3f} ms "
          f"= {100*floor/whole:.2f}% of the step")
    if negs:
        worst = min(negs, key=lambda nd: nd[1])
        if floor > whole * a.additivity_tol / 100:
            raise SystemExit(
                f"\nREFUSING TO RANK: {len(negs)} stages measured NEGATIVE "
                f"(worst {worst[0]} at {worst[1]:.2f} ms). A longer prefix "
                "cannot be faster than a shorter one, so run-to-run noise "
                "exceeds the stages being measured. Raise --reps, quiet the "
                "box, or raise --context so each stage is larger than the "
                "noise. Additivity can PASS while this fails -- the errors "
                "cancel in the sum -- so this is the binding check.")
        print(f"  WARNING: {len(negs)} negative deltas (worst {worst[0]} "
              f"{worst[1]:.2f} ms); stages near the noise floor are not "
              "trustworthy individually.")
    print()

    agg = {}
    for name, kind, d in rows:
        key = kind or name
        agg[key] = agg.get(key, 0.0) + d
    print("  BY STAGE TYPE (this is the ranking to act on)")
    for key, tot in sorted(agg.items(), key=lambda kv: -kv[1]):
        print(f"    {key:14s} {tot:8.2f} ms   {100*tot/whole:5.1f}%")

    print(f"\n  TOP {a.top} INDIVIDUAL STAGES")
    for name, kind, d in sorted(rows, key=lambda r: -r[2])[: a.top]:
        print(f"    {name:16s} {d:8.3f} ms   {100*d/whole:5.2f}%")
    print(f"\n  contention gate PASSED: {w0:.1f} W -> {w1:.1f} W\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
