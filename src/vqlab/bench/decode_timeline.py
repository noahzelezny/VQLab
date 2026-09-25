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

AND IT MEASURES WHETHER THAT IS TRUE, rather than assuming it. Prefix
deltas are only a PARTITION if the stages run strictly serially. At decode
N=1 the data dependencies are serial (layer i+1 needs layer i), but that is
an argument, not a measurement: MLX submits asynchronously, and kernels or
host work from adjacent stages may overlap. So the sum of the deltas is
compared against an independently measured full step and the drift is
REPORTED, with its sign, as the serial-execution check:

    sum > whole   either each prefix pays its own flush/tail that the whole
                  run pays once, or the stages genuinely OVERLAP in the full
                  step. Both push this way; the deltas are upper bounds
    sum ~ whole   serial, and the deltas are a clean partition
    sum < whole   the parts UNDER-ACCOUNT. This is NOT overlap -- overlap
                  makes the whole FASTER than its parts (sum > whole), not
                  slower. Expect estimator bias: a paired delta cancels a
                  noise bias common to both halves, a single absolute whole
                  does not. More reps should shrink it

None of these is a failure. Only the third invalidates "stage X costs Y ms",
and it is still a valid ranking. The tool does NOT refuse on drift by
default; `--require-clean` restores a hard gate for when a strict partition
is actually needed.

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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # src/
from vqlab import _layout  # noqa: E402,F401  stage dirs -> sys.path
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
                    help="percent drift treated as 'clean serial'. Reported "
                         "either way; only --require-clean makes it fatal.")
    ap.add_argument("--require-clean", action="store_true",
                    help="refuse to rank unless the deltas sum to the whole "
                         "and no delta is negative. OFF by default: a "
                         "partition that does not add up is still a valid "
                         "RANKING, and overlap is physics, not a defect.")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--json-out", default=None,
                    help="write the shares and diagnostics as JSON, so a "
                         "campaign can be compared without re-running it")
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
        # gdn and full attention are SEPARATE stage types. Lumping them
        # hides the thing most worth knowing: GatedDeltaNet carries
        # fixed-size recurrent state while full attention grows with
        # context, so their shares move in opposite directions as context
        # grows and a combined "attn" number is a blend of the two at one
        # context length.
        kind = "gdn" if getattr(lyr, "is_linear", False) else "attn"
        stages.append((f"L{i:02d}.{kind}", i, kind))
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

    def timed_pair(k):
        """Time prefix k-1 and prefix k INTERLEAVED in one window.

        WHY PAIRED. Taking cum[k-1] and cum[k] as independent best-of-N and
        subtracting compares two absolutes captured minutes apart, so any
        drift between them lands entirely in the delta. Under load that
        drift dwarfs a stage: the first contended run of this instrument
        (155 W) produced 59 negative deltas, a noise floor of 72% of the
        step, and an `attn` share of -261%. Interleaving k-1 and k inside
        one window is the same paired principle the KL work uses -- both
        halves see the same machine state, and slow drift cancels in the
        difference instead of accumulating into it.
        """
        c = fresh_cache()
        best_a = best_b = float("inf")
        for _ in range(a.reps + 1):
            for which in (0, 1):
                kk = (k - 1) if which == 0 else k
                mx.synchronize()
                t0 = time.time()
                out = forward_prefix(kk, c, tokid)
                mx.eval(out)
                mx.synchronize()
                dt = time.time() - t0
                if which == 0:
                    best_a = min(best_a, dt)
                else:
                    best_b = min(best_b, dt)
        del c
        return (best_b - best_a) * 1e3, best_b * 1e3

    def timed(k):
        # ONE cache rebuild per stage, reps inside it. Rebuilding per rep
        # made the prefill dominate the run (4x the prefills for the same
        # number of timings). The bias this accepts: each rep leaves one
        # extra token in the cache, so rep n sees a context of
        # `context + n - 1`. At the default 64 that is a <5% context drift
        # across 4 reps, and only the full-attention layers notice it at
        # all -- the GatedDeltaNet layers carry fixed-size recurrent state.
        # `best-of` then takes the fastest, which is the SHORTEST context,
        # so the bias does not accumulate into the reported number.
        c = fresh_cache()
        best = float("inf")
        for _ in range(a.reps + 1):
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

    rows = []
    for k, (name, _li, kind) in enumerate(stages):
        if k == 0:
            d = t = timed(0)
        else:
            d, t = timed_pair(k)
        rows.append((name, kind, d))
        if k % 16 == 0:
            print(f"  [{k:3d}/{len(stages)}] {name:16s} "
                  f"stage {d:7.3f} ms  (cum {t:8.2f})", flush=True)

    whole = timed(len(stages) - 1)

    # PAIRED WHOLE -- the control that decides whether a negative drift is
    # real OVERLAP or just an estimator mismatch. The per-stage deltas come
    # from timed_pair (two prefixes interleaved in one window); `whole`
    # above comes from timed (one prefix, best-of-N, its own window). With
    # paired measurement the deltas do NOT telescope -- each pair has its
    # own cache and window -- so comparing their sum against an UNPAIRED
    # whole compares two different estimators, and any bias between them
    # lands in the drift and looks exactly like overlap.
    # Measuring (last prefix - first prefix) through timed_pair gives the
    # whole step MINUS embed under the SAME estimator as the deltas:
    #   whole_paired ~ sum(deltas)  -> the gap is ESTIMATOR BIAS, not overlap
    #   whole_paired ~ whole        -> the gap is REAL: stages overlap
    def timed_pair_span(lo, hi):
        c = fresh_cache()
        ba = bb = float("inf")
        for _ in range(a.reps + 1):
            for which, kk in ((0, lo), (1, hi)):
                mx.synchronize()
                t0 = time.time()
                out = forward_prefix(kk, c, tokid)
                mx.eval(out)
                mx.synchronize()
                dt = time.time() - t0
                if which == 0:
                    ba = min(ba, dt)
                else:
                    bb = min(bb, dt)
        del c
        return (bb - ba) * 1e3

    span = timed_pair_span(0, len(stages) - 1)
    whole_paired = span + rows[0][2]     # + embed, which rows[0] carries
    total = sum(d for _, _, d in rows)
    drift = abs(total - whole) / whole * 100

    w1 = gpu_watts()
    if w1 > a.max_watts:
        raise SystemExit(
            f"\nVOID: GPU ended at {w1:.1f} W (started {w0:.1f}). A foreign "
            "job landed mid-run; discard every number above.")

    print(f"\n  full step (unpaired)     {whole:8.2f} ms")
    print(f"  full step (PAIRED span)  {whole_paired:8.2f} ms"
          f"   <- same estimator as the deltas")
    print(f"  sum of stage deltas      {total:8.2f} ms   drift {drift:.2f}%")
    floor_pre = -min((d for _n, _k, d in rows), default=0.0)
    noise_frac = floor_pre / whole * 100
    signed = (total - whole) / whole * 100
    signed_p = (total - whole_paired) / whole_paired * 100
    est_gap = (whole_paired - whole) / whole * 100
    if abs(signed) <= a.additivity_tol:
        verdict = ("SERIAL: deltas are a clean partition; a stage's number "
                   "is its standalone cost")
    elif noise_frac > 10.0:
        verdict = (f"UNDETERMINED: the worst negative delta is "
                   f"{noise_frac:.0f}% of the step, so run-to-run noise is "
                   "larger than most stages and NOTHING can be concluded "
                   "about serial-vs-overlap from this drift. Noise is "
                   "checked FIRST on purpose: an earlier version reported "
                   "'OVERLAP' for what was simply a contended box.")
    elif signed > 0:
        verdict = (f"SERIAL + per-measurement tax: the prefixes cost "
                   f"{signed:+.1f}% more than the whole, i.e. each prefix "
                   f"pays a flush the whole run pays once (~"
                   f"{(total-whole)/max(len(rows),1):.3f} ms/stage). Stage "
                   "costs are UPPER bounds; the RANKING is unaffected")
    elif abs(signed_p) <= a.additivity_tol:
        verdict = (f"ESTIMATOR GAP, NOT OVERLAP: against the UNPAIRED whole "
                   f"the drift is {signed:+.1f}%, but against the PAIRED "
                   f"whole (same estimator as the deltas) it is only "
                   f"{signed_p:+.1f}%. The paired and unpaired estimators "
                   f"differ by {est_gap:+.1f}%; that difference was "
                   "masquerading as overlap. Deltas ARE a clean partition")
    elif signed < 0:
        verdict = (f"PARTS UNDER-ACCOUNT by {-signed:.1f}%: the sum of the "
                   "deltas is LESS than the whole. NOTE THE SIGN -- this is "
                   "NOT overlap. Concurrency would make the whole FASTER "
                   "than the sum of its parts, i.e. sum > whole. sum < whole "
                   "means the parts miss time the whole pays, and the "
                   "expected cause is estimator bias, not physics: each "
                   "PAIRED delta is a difference taken in ONE window, so a "
                   "positive noise bias common to both halves CANCELS, while "
                   "the whole is a single absolute measurement that KEEPS "
                   "its bias. The deltas are then the more trustworthy "
                   "number and the whole is the inflated one. Raise --reps "
                   "and this gap should shrink toward zero; if it does not, "
                   "look for real missing work (something the full forward "
                   "does that forward_prefix does not)")
    else:
        verdict = (f"UNEXPLAINED at {signed:+.1f}%")
    print(f"  serial-execution check: {verdict}")
    if a.require_clean and abs(signed) > a.additivity_tol:
        raise SystemExit("\nREFUSING (--require-clean): drift exceeds "
                         f"{a.additivity_tol}%.")

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
        msg = (f"  {len(negs)} NEGATIVE deltas (worst {worst[0]} "
               f"{worst[1]:.2f} ms). A longer prefix cannot truly be faster "
               "than a shorter one, so this is run-to-run noise (or "
               "scheduling overlap). Stages below the noise floor above are "
               "not trustworthy INDIVIDUALLY; aggregates over many stages "
               "still are, because the noise is zero-mean and cancels.")
        if a.require_clean and floor > whole * a.additivity_tol / 100:
            raise SystemExit("\nREFUSING (--require-clean):\n" + msg)
        print(msg)
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
    if a.json_out:
        pathlib.Path(a.json_out).write_text(json.dumps({
            "artifact": str(a.art), "context": a.context, "reps": a.reps,
            "whole_ms": whole, "whole_paired_ms": whole_paired,
            "sum_deltas_ms": total, "drift_pct": signed,
            "drift_vs_paired_pct": signed_p,
            "noise_floor_ms": floor, "noise_floor_pct": noise_frac,
            "n_negative": len(negs), "verdict": verdict,
            "watts": [w0, w1],
            "shares": {k: {"ms": v, "pct": 100 * v / whole}
                       for k, v in agg.items()},
            "stages": [{"name": n, "kind": k, "ms": d} for n, k, d in rows],
        }, indent=1))
        print(f"  -> {a.json_out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
