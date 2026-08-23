"""moemash price — size-target a VQ build BEFORE fitting it.

Two measured size models (see the paper, §2.4/§3.3):

  MoE harvest form (measured on Qwen3.5-397B-A17B, 6-for-7 within ±0.4 GiB):
      new_size = base_size − shallow_gib_per_bit × shallow_bits_harvested
    where shallow_gib_per_bit is the byte mass of the harvestable shallow
    band per bit of code width (1.87 GiB/bit for the 397B, L0–9).

  Dense composition form (closed to ≤0.003 GiB, three builds, two geometries):
      total = codes + scales + carry,   codes = params/d × bits/8
    with measured constants for the dense 27B: scales = 0.498 GiB,
    non-MLP carry = 5.129 GiB.

Everything here is an ESTIMATE for planning which fit to run; the measured
packed size of the artifact you actually build is the only citable size
(METHODOLOGY.md §2). Constants are per-family MEASURED values; families not
listed need one profiling pass (docs/ONBOARDING.md §1) to fill them in.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass

GIB = 1024**3


@dataclass
class MoEFamily:
    name: str
    shallow_gib_per_bit: float          # GiB per bit of shallow code width
    body_gib_per_bit: float             # GiB per bit of body code width
    flat_rungs: dict[str, float]        # label -> measured packed GiB (post-graft)


@dataclass
class DenseFamily:
    name: str
    mlp_params: float                   # params covered by VQ (the MLP trio)
    scales_gib: float                   # measured
    carry_gib: float                    # measured non-MLP carry


# Measured constants. 397B rung sizes are measured packed post-graft bytes
# (paper table 3.1); the two-coefficient bands come from the harvest fits.
MOE_FAMILIES = {
    "qwen397b": MoEFamily(
        name="Qwen3.5-397B-A17B",
        shallow_gib_per_bit=1.87,
        body_gib_per_bit=8.81,
        flat_rungs={
            "d8/K16384": 100.97,
            "d4/K128": 100.93,
            "d4/K256": 111.62,
            "d4/K512": 122.31,
            "d4/K2048": 143.68,
        },
    ),
}

DENSE_FAMILIES = {
    # 27B dense: codes = params/d * bits/8; +0.498 scales +5.129 carry.
    "dense27b": DenseFamily(
        name="dense 27B",
        mlp_params=None,  # derived below from a measured rung instead
        scales_gib=0.498,
        carry_gib=5.129,
    ),
}

# One measured anchor rung lets us derive the dense codes mass without
# shipping a parameter count: d2/K256 (4.00 bpw over the MLP trio) measured
# 13.596 GiB packed => codes = 13.596 - 0.498 - 5.129 = 7.969 GiB at 4 bpw.
DENSE27B_CODES_GIB_PER_BPW = (13.596 - 0.498 - 5.129) / 4.0


def dense_total_gib(bpw: float, fam: DenseFamily) -> float:
    return DENSE27B_CODES_GIB_PER_BPW * bpw + fam.scales_gib + fam.carry_gib


def price_moe(fam: MoEFamily, budget_gib: float) -> str:
    lines = [f"family: {fam.name}", f"budget: {budget_gib:.2f} GiB", ""]
    # Closest flat rung at or under budget, and harvest recipes from richer bases.
    rungs = sorted(fam.flat_rungs.items(), key=lambda kv: kv[1])
    under = [(l, s) for l, s in rungs if s <= budget_gib]
    over = [(l, s) for l, s in rungs if s > budget_gib]
    if under:
        l, s = under[-1]
        lines.append(f"best flat rung at-or-under budget: {l} = {s:.2f} GiB "
                     f"(headroom {budget_gib - s:.2f} GiB)")
    else:
        lines.append("no measured flat rung fits this budget; smallest is "
                     f"{rungs[0][0]} = {rungs[0][1]:.2f} GiB")
    for l, s in over:
        bits = (s - budget_gib) / fam.shallow_gib_per_bit
        if 0 < bits <= 8:
            lines.append(
                f"harvest recipe from {l} ({s:.2f} GiB): shed "
                f"{bits:.2f} shallow bits -> predicted {budget_gib:.2f} GiB "
                f"(model: new = base − {fam.shallow_gib_per_bit} GiB × bits)"
            )
    lines += [
        "",
        "notes: harvest is ~2x the byte-efficiency of stepping down a flat",
        "rung, but never beats a flat rung at the flat rung's own size —",
        "if a flat rung sits exactly at your budget, fit that instead.",
        "Prediction bands: ±0.4 GiB (measured, 6-for-7). Sizes post-graft.",
    ]
    return "\n".join(lines)


def price_dense(fam: DenseFamily, budget_gib: float) -> str:
    lines = [f"family: {fam.name}", f"budget: {budget_gib:.2f} GiB", ""]
    bpw = (budget_gib - fam.scales_gib - fam.carry_gib) / DENSE27B_CODES_GIB_PER_BPW
    lines.append(f"model: total = codes + {fam.scales_gib} + {fam.carry_gib} GiB")
    if bpw <= 0:
        lines.append("budget is below the fixed carry+scales mass; not reachable.")
        return "\n".join(lines)
    lines.append(f"MLP bpw at budget: {bpw:.3f}")
    # Enumerate (d, K) geometries whose rate log2(K)/d lands at or under bpw.
    lines.append("geometries at or under budget (rate = log2(K)/d):")
    best = []
    for d in (2, 4, 8):
        for logk in range(4, 17):
            rate = logk / d
            if rate <= bpw + 1e-9:
                best.append((rate, d, 2**logk))
    best.sort(reverse=True)
    for rate, d, K in best[:6]:
        tg = K * d * 2
        kernel = "threadgroup" if tg < 32768 else "device-memory codebook"
        lines.append(
            f"  d{d}/K{K}: {rate:.2f} bpw -> predicted "
            f"{dense_total_gib(rate, fam):.3f} GiB  [{kernel} kernel]"
        )
    lines += [
        "",
        "notes: at matched rate, larger d has measured better (~8–12% KL,",
        "two bands) — prefer the highest-K geometry your budget allows,",
        "then larger d among rate twins. Prediction error ≤0.003 GiB",
        "(measured, dense 27B). Constants are dense-27B; other dense",
        "families need one profiling pass to refit them.",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="moemash price",
        description="Price a VQ build before fitting it: name a byte budget, "
                    "get the candidate recipes.",
    )
    ap.add_argument("--family", required=True,
                    choices=sorted(MOE_FAMILIES) + sorted(DENSE_FAMILIES))
    ap.add_argument("--budget-gib", type=float, required=True)
    args = ap.parse_args(argv)
    if args.family in MOE_FAMILIES:
        print(price_moe(MOE_FAMILIES[args.family], args.budget_gib))
    else:
        print(price_dense(DENSE_FAMILIES[args.family], args.budget_gib))
    return 0


if __name__ == "__main__":
    sys.exit(main())
