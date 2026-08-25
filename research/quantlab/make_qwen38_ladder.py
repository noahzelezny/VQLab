#!/usr/bin/env python3
"""qwen38_ladder.png for the Qwen3.8-27B card lineup.

All sizes measured post-graft (333-tensor bf16 vision tower, 0.859 GiB),
rungs and comparators alike. The affine rungs are our own conversions:
no MLX quantization of this model has been published by the community.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REL = [(12.468, 85.8, "VQ-3.9bpw"), (14.454, 40.3, "VQ-4.5bpw"),
       (15.450, 32.8, "VQ-4.8bpw")]
LAB = [(10.558, 325.6, None), (11.468, 148.5, None), (18.441, 26.7, None)]
AFF = [(8.690, 1426.9, "q2"), (11.821, 187.8, "q3"), (14.952, 45.8, "q4"),
       (21.214, 3.71, "q6"), (27.475, 1.254, "q8")]

fig, ax = plt.subplots(figsize=(7.6, 4.6))
allvq = sorted(REL + LAB)
ax.plot([s for s, _, _ in allvq], [k for _, k, _ in allvq], "-", color="#1f77b4",
        lw=1.4, alpha=0.5, zorder=2)
ax.plot([s for s, _, _ in LAB], [k for _, k, _ in LAB], "o", color="#1f77b4",
        ms=6, mfc="white", label="VQ (measured, unreleased)", zorder=3)
ax.plot([s for s, _, _ in REL], [k for _, k, _ in REL], "o", color="#1f77b4",
        ms=9, label="VQ (released)", zorder=4)
ax.plot([s for s, _, _ in AFF], [k for _, k, _ in AFF], "s--", color="#d62728",
        lw=1.8, ms=7, label="affine (our conversions)", zorder=3)
for s, k, n in REL:
    ax.annotate(n, (s, k), textcoords="offset points", xytext=(0, 12),
                ha="center", fontsize=8, color="#1f77b4")
for s, k, n in AFF:
    if n:
        ax.annotate(n, (s, k), textcoords="offset points", xytext=(0, -15),
                    ha="center", fontsize=8, color="#d62728")
ax.axvspan(16.5, 21.5, color="#d62728", alpha=0.06, zorder=1)
ax.annotate("affine wins above ~5 bpw", (19.0, 700), ha="center", fontsize=8,
            color="#d62728", alpha=0.85)
ax.set_yscale("log")
ax.set_xlabel("artifact size (GiB, incl. bf16 vision tower)")
ax.set_ylabel("KL to bf16  (millinats/token, log scale)")
ax.set_title("Qwen3.8-27B — divergence vs size (lower is better)")
ax.grid(alpha=0.3, which="both")
ax.legend(frameon=False, loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig("qwen38_ladder.png", dpi=200)
print("wrote qwen38_ladder.png")
