"""Paired draft-acceptance comparison across independent prompts.

    python -m vqlab.cli mtp-accept --model <artifact>
        --head q6=<sidecar> [--head e4q8=<sidecar> ...]
        [--aligns committed,legacy] [--tokens 256] [--out rec.json]

This is the RELIABLE instrument in this package, and `mtp-bench` is not.
Acceptance is a property of the model and the head; greedy decoding is
deterministic, so every figure reproduces exactly and nothing here is
perturbed by machine state. Wall-clock is a property of the machine: on a
thermally constrained box the same configuration measured twice in one
process gave 1.723x and 1.135x (2026-08-31, M4 Max). Quote speedups from a
thermally stable machine; quote acceptance from here.

Two design points, both learned the hard way:

  Repeats buy nothing. Re-running a greedy generation reproduces the previous
  acceptance to four decimals, because the trajectory is deterministic. Extra
  replicates of the SAME prompt add zero information.

  Prompts are the replicates, and the comparison is paired. Treating one
  trajectory's N steps as N independent Bernoulli trials overstates the power
  badly: the steps share a prefix, so they are correlated and the effective
  sample size is far below the nominal one. Independent prompts are genuine
  replicates, and running every prompt through every configuration removes
  prompt difficulty -- by far the largest source of spread -- from the
  comparison.
"""
import argparse
import json
import pathlib
import statistics
import sys

import mlx.core as mx

DEFAULT_PROMPTS = [
    "Explain why vector quantization compresses neural network weights better than scalar rounding.",
    "Write a Python function that merges two sorted lists without using sorted().",
    "What were the main causes of the 1973 oil crisis?",
    "Summarise the tradeoffs between mutexes and channels for concurrency.",
    "Translate to French: 'The weather tomorrow will be cold and clear.'",
    "Derive the variance of the sample mean for i.i.d. observations.",
    "Describe how a B-tree differs from a binary search tree, and when each wins.",
    "Write a haiku about a lighthouse in winter.",
    "Given a REST API returning 429s intermittently, outline a debugging plan.",
    "What is the difference between an eigenvalue and a singular value?",
    "Explain the halting problem to a first-year undergraduate.",
    "List the steps to safely rotate a production database credential.",
]


def paired_stats(rows, heads, aligns, n_prompts):
    """Per-head paired delta between the two alignment schemes."""
    out = []
    for label in heads:
        d = []
        for pi in range(n_prompts):
            g = {r["align"]: r for r in rows
                 if r["head"] == label and r["prompt"] == pi}
            if len(g) == 2:
                d.append(g[aligns[0]]["acceptance"] - g[aligns[1]]["acceptance"])
        if len(d) > 1:
            sd = statistics.stdev(d)
            se = sd / len(d) ** 0.5
            m = statistics.mean(d)
            out.append({"head": label, "mean_delta": m, "se": se,
                        "t": (m / se) if se else float("nan"), "n": len(d),
                        "wins": sum(1 for x in d if x > 0),
                        "deltas": [round(x, 5) for x in d]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--head", action="append", default=[],
                    help="label=sidecar path, repeatable")
    ap.add_argument("--family", default=None)
    ap.add_argument("--aligns", default="committed")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--prompts", default=None,
                    help="JSON file with a list of prompt strings")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from mlx_lm.utils import load
    from vqlab.mtp import load_mtp_head, mtp_stream_generate

    prompts = (json.loads(pathlib.Path(a.prompts).read_text())
               if a.prompts else DEFAULT_PROMPTS)
    aligns = a.aligns.split(",")

    model, tok = load(a.model, lazy=False, trust_remote_code=True)
    heads = {}
    for spec in a.head:
        label, _, path = spec.partition("=")
        heads[label], _ = load_mtp_head(model, sidecar=path, family=a.family)
        mx.clear_cache()
    if not heads:
        raise SystemExit("pass at least one --head label=path")
    print(f"resident {mx.get_active_memory()/2**30:.2f} GiB | "
          f"{len(heads)} heads x {len(aligns)} schemes x {len(prompts)} prompts",
          flush=True)

    rows = []
    for pi, text in enumerate(prompts):
        ids = tok.apply_chat_template([{"role": "user", "content": text}],
                                      add_generation_prompt=True)
        for label, head in heads.items():
            for align in aligns:
                last = None
                for last in mtp_stream_generate(model, tok, ids, head,
                                                max_tokens=a.tokens,
                                                align=align, family=a.family):
                    pass
                rows.append({"prompt": pi, "head": label, "align": align,
                             "acceptance": last.acceptance, "steps": last.steps,
                             "accepted": last.accepted,
                             "tokens": last.generation_tokens})
                print(f"  p{pi:02d} {label:6s} {align:9s} "
                      f"acc {last.acceptance:.4f} "
                      f"({last.accepted}/{last.steps})", flush=True)
        if a.out:
            pathlib.Path(a.out).write_text(json.dumps(rows, indent=1))

    print("\npooled over prompts:", flush=True)
    for label in heads:
        for align in aligns:
            rs = [r for r in rows if r["head"] == label and r["align"] == align]
            acc = sum(r["accepted"] for r in rs) / sum(r["steps"] for r in rs)
            print(f"  {label:6s} {align:9s} {acc:.4f}  "
                  f"({sum(r['steps'] for r in rs)} steps over {len(rs)} prompts)",
                  flush=True)

    if len(aligns) == 2:
        print(f"\npaired {aligns[0]} - {aligns[1]}, prompts as replicates:",
              flush=True)
        for st in paired_stats(rows, heads, aligns, len(prompts)):
            print(f"  {st['head']:6s} {st['mean_delta']*100:+.2f}pp  "
                  f"SE {st['se']*100:.2f}pp  t={st['t']:+.2f}  n={st['n']}  "
                  f"better on {st['wins']}/{st['n']} prompts", flush=True)
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
