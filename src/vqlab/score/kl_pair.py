#!/usr/bin/env python
"""kl-pair — paired KL comparison between two arms scored in SEPARATE runs.

`kl-ladder` pairs rungs WITHIN one invocation, against the first --rung as
reference. That cannot express an arm that is an ENVIRONMENT VARIABLE rather
than a directory: VQ_DENSE_SS, VQ_D4_WALK, the DEVX twins all change the same
artifact in place, and score_one inherits the parent environment, so the two
arms must be two invocations. This pairs their per-position arrays after the
fact, using kl_ladder's own formula so the test is the same test.

WHY PAIRED. Both arms saw the SAME positions and the SAME teacher, so
differencing per position removes the position-to-position variance that
dominates each arm's own SEM (typically 6 mnats here against a paired SEM
near 0.01). Comparing two overlapping 95% intervals is NOT this test and is
far more conservative -- an overlap there does not mean "no difference".

READ THE MAGNITUDE, NOT ONLY THE SIGNIFICANCE. At n=12288 with a
deterministic difference, |t| will exceed 2 for an arbitrarily small real
shift. A significant delta of 0.1 mnats against a rung sitting at 140 is not
a quality event. This prints the delta as a PERCENTAGE of the reference arm's
own KL for exactly that reason.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from safetensors.numpy import load_file


def paired(fa: str, fb: str) -> dict:
    """kl_ladder.paired(), applied across invocations. arm minus reference."""
    x = load_file(fa)["kl_millinats"].astype(np.float64)
    y = load_file(fb)["kl_millinats"].astype(np.float64)
    if x.shape != y.shape:
        raise SystemExit(f"FAIL: shape mismatch {x.shape} vs {y.shape} — "
                         "these arms did not see the same positions, so they "
                         "cannot be paired.")
    d = x - y
    n = d.size
    sem = float(d.std(ddof=1) / np.sqrt(n))
    m = float(d.mean())
    return {"delta": m, "sem": sem, "t": (m / sem) if sem else 0.0, "n": n,
            "ref_mean": float(y.mean()), "arm_mean": float(x.mean())}


def verify(fa: str, fb: str, allow: bool = False) -> list:
    """Refuse a pairing the sidecars cannot vouch for; return the problems.

    Two arrays pair only if they hold the SAME positions against the SAME
    teacher, computed by the SAME numerics build. File names say none of
    that; the .json sidecar stream_score / kl-ladder write beside every array
    does (cache identity, build stamp, position count)."""
    from vqlab.core.numerics import pairing_problems

    def side(f):
        try:
            return json.load(open(f + ".json"))
        except (OSError, ValueError):
            return None
    probs = pairing_problems(side(fa), side(fb))
    if probs and not allow:
        raise SystemExit(f"FAIL: cannot verify that {os.path.basename(fa)} and "
                         f"{os.path.basename(fb)} are pairable: " + "; ".join(probs)
                         + ". Pass --allow-unverified-pair to pair them anyway "
                         "(the output says so).")
    return probs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--per-pos-dir", required=True)
    ap.add_argument("--arm", required=True, help="rung name of the ARM")
    ap.add_argument("--ref", required=True, help="rung name of the REFERENCE")
    ap.add_argument("--corpus", action="append", required=True,
                    help="corpus name, repeatable (must match kl-ladder's)")
    ap.add_argument("--allow-unverified-pair", action="store_true",
                    help="pair arrays whose sidecars do not prove the same cache and "
                         "numerics build (old arrays, or a deliberate cross-build "
                         "comparison); every such corpus is flagged UNVERIFIED")
    a = ap.parse_args()

    print(f"\npaired KL: {a.arm} minus {a.ref}   (negative = arm is BETTER)\n")
    print(f"  {'corpus':<10} {'ref KL':>10} {'arm KL':>10} {'delta':>9} "
          f"{'%':>7} {'sem':>8} {'t':>8}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*9} {'-'*7} {'-'*8} {'-'*8}")
    worst = 0.0
    unverified = []
    for c in a.corpus:
        fa = os.path.join(a.per_pos_dir, f"{a.arm}__{c}.safetensors")
        fb = os.path.join(a.per_pos_dir, f"{a.ref}__{c}.safetensors")
        for f in (fa, fb):
            if not os.path.exists(f):
                raise SystemExit(f"FAIL: missing {f}")
        probs = verify(fa, fb, a.allow_unverified_pair)
        if probs:
            unverified.append((c, probs))
        r = paired(fa, fb)
        pct = r["delta"] / r["ref_mean"] * 100 if r["ref_mean"] else 0.0
        worst = max(worst, abs(pct))
        print(f"  {c:<10} {r['ref_mean']:>10.3f} {r['arm_mean']:>10.3f} "
              f"{r['delta']:>+9.3f} {pct:>+6.2f}% {r['sem']:>8.4f} "
              f"{r['t']:>+8.2f}")
    print(f"\n  n = {r['n']} positions per corpus. |t|>2 is the F118 gate, but "
          f"at this n\n  it fires on trivial shifts -- the worst magnitude here "
          f"is {worst:.2f}% of\n  the reference arm's own KL.\n")
    if unverified:
        print("  UNVERIFIED PAIRING (--allow-unverified-pair):")
        for c, probs in unverified:
            print(f"    {c}: " + "; ".join(probs))
    else:
        print("  pairing verified from sidecars: same cache, same numerics build.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
