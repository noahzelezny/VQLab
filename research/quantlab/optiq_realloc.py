#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""Re-allocate bits from an EXISTING optiq sensitivity checkpoint.

WHY. `optiq convert --method optiq` spends ~10.6h on Qwen3.8-27B profiling
per-component KL sensitivity (326 components x 4 bit choices x n samples,
~0.5 components/min on the M3 Ultra). That cost buys the SENSITIVITY, which
is a property of the model and the calibration mix — not of the budget.

Bit ALLOCATION is a separate, cheap step: `optimize_mixed_precision` reads
the checkpoint and runs a greedy knapsack. `OPTIQ_ATTN_FLOOR_BITS` is
applied there too (patches/moe-allocator-fixes.patch touches
core/optimizer.py, not core/sensitivity.py). So every target-bpw and every
attention floor can be explored for free once the checkpoint exists.

This is the same move README.md's verify snippet makes.

    ./optiq_realloc.py --checkpoint <dir>/sensitivity_checkpoint.json \
        --bpw 2.8 3.0 3.2 --attn-floor 0 4 6

Prints the achieved bpw and the bit histogram for each combination so a
budget can be picked BEFORE paying for a conversion.
"""
import argparse
import collections
import json
import os
import pathlib


def load_results(path):
    from optiq.core.sensitivity import SensitivityResult
    d = json.load(open(path))
    return [SensitivityResult(
        layer_name=e["layer_name"],
        sensitivities={int(k): v for k, v in e["sensitivities"].items()},
        param_count=e["param_count"]) for e in d]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--bpw", nargs="+", type=float, default=[3.0])
    ap.add_argument("--attn-floor", nargs="+", type=int, default=[0],
                    help="0 = no floor (let calibration decide)")
    ap.add_argument("--candidate-bits", default="2,3,4,6")
    args = ap.parse_args()

    bits = [int(b) for b in args.candidate_bits.split(",")]
    n = len(json.load(open(args.checkpoint)))
    print(f"checkpoint: {n} components profiled\n")

    print(f"{'attn_floor':>10} {'target':>7} {'achieved':>9}   bit histogram")
    for floor in args.attn_floor:
        # Must be set BEFORE importing/reloading the optimizer module, since
        # the patch reads it at allocation time.
        if floor:
            os.environ["OPTIQ_ATTN_FLOOR_BITS"] = str(floor)
        else:
            os.environ.pop("OPTIQ_ATTN_FLOOR_BITS", None)
        import importlib
        from optiq.core import optimizer as opt
        importlib.reload(opt)
        results = load_results(args.checkpoint)
        for target in args.bpw:
            alloc = opt.optimize_mixed_precision(
                results, target_bpw=target, candidate_bits=bits)
            hist = collections.Counter(alloc.bit_allocation.values()) \
                if hasattr(alloc, "bit_allocation") else {}
            print(f"{floor or '-':>10} {target:>7.2f} "
                  f"{alloc.achieved_bpw:>9.4f}   {dict(sorted(hist.items()))}")

    print("\nPick a (floor, target) pair, then convert with those settings — "
          "the expensive sensitivity pass is already paid for.")


if __name__ == "__main__":
    main()
