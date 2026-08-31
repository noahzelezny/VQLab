"""Measure the MTP drafting speedup, acceptance, and numerics control.

    python -m vqlab.cli mtp-bench --model <artifact> [--sidecar <file>]
        [--tokens 128] [--out record.json]

Decodes the same prompt twice — speculative and plain single-token — after
warming both kernel paths, and at temperature 0 also runs the chunk control.
Read `vqlab/mtp/bench.py` for what each number means before quoting one;
METHODOLOGY.md is the contract for publishing any of them.
"""
import argparse
import json
import pathlib
import sys

from vqlab.mtp.bench import benchmark
from vqlab.mtp_run import add_sampling_args, load_all, prompt_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sidecar", default=None)
    ap.add_argument("--family", default=None)
    ap.add_argument("--prompt", default="Explain why vector quantization "
                    "compresses neural network weights better than scalar "
                    "rounding.")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ab", action="store_true",
                    help="run both head-cache schemes and report the delta")
    add_sampling_args(ap)
    a = ap.parse_args()

    model, tok, head = load_all(a)
    schemes = ["committed", "legacy"] if a.ab else [a.align]
    recs = []
    for scheme in schemes:
        rec = benchmark(model, tok, prompt_ids(tok, a.prompt), head,
                        tokens=a.tokens, temp=a.temp, warmup=a.warmup,
                        family=a.family, align=scheme)
        rec["model"], rec["align"] = a.model, scheme
        recs.append(rec)
    rec = recs[0]

    print(f"\n{rec.pop('text')}\n", flush=True)
    for r in recs[1:]:
        r.pop("text", None)
    print(f"baseline:    {rec['baseline_tok_s']:.2f} tok/s", flush=True)
    for r in recs:
        print(f"speculative [{r['align']:9s}]: {r['speculative_tok_s']:.2f} "
              f"tok/s  {r['speedup']:.2f}x  ({r['steps']} steps, acceptance "
              f"{r['acceptance']:.3f})", flush=True)
    print(f"SPEEDUP: {rec['speedup']:.2f}x", flush=True)
    if "chunk_control_disagreements" in rec:
        n = rec["chunk_control_disagreements"]
        print(f"chunk control: chunked vs single-token greedy disagree at "
              f"{n}/{rec['chunk_control_positions']} positions"
              + (f"; top-2 logit gaps there {rec['chunk_control_gaps']} vs "
                 f"median {rec['chunk_control_median_gap']}" if n else ""),
              flush=True)
    print(json.dumps(recs if len(recs) > 1 else rec), flush=True)
    if a.out:
        pathlib.Path(a.out).write_text(
            json.dumps(recs if len(recs) > 1 else rec, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
