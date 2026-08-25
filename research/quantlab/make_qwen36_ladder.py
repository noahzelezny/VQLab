#!/usr/bin/env python3
"""Regenerate qwen36_ladder.png for the Qwen3.6-35B-A3B card lineup.

No generator was kept for the 08-19 original, which is how it went stale:
it predates both the 08-24 vision grafts (+0.832 GiB on every rung) and two
of the four releases. Keeping this file means the next size change is a
one-line edit rather than an archaeology problem.

All sizes are measured, post-graft, and include the bf16 vision tower that
the community comparators also carry.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REL = [(13.809, 85.535, "VQ-3.4bpw"), (15.670, 53.022, "VQ-3.8bpw"),
       (18.729, 44.573, "VQ-4.6bpw"), (22.226, 28.141, "VQ-5.4bpw")]
AFF = [(19.000, 78.557, "mlx 4-bit"), (35.131, 7.449, "mlx 8-bit")]

fig, ax = plt.subplots(figsize=(7.6, 4.6))
ax.plot([s for s, _, _ in REL], [k for _, k, _ in REL], "o-", color="#1f77b4",
        lw=2, ms=7, label="VQ (this collection)", zorder=3)
ax.plot([s for s, _, _ in AFF], [k for _, k, _ in AFF], "s--", color="#d62728",
        lw=1.8, ms=7, label="affine (mlx-community)", zorder=3)
for s, k, n in REL:
    ax.annotate(n, (s, k), textcoords="offset points", xytext=(0, 11),
                ha="center", fontsize=8, color="#1f77b4")
for s, k, n in AFF:
    ax.annotate(n, (s, k), textcoords="offset points", xytext=(0, -16),
                ha="center", fontsize=8, color="#d62728")
ax.set_yscale("log")
ax.set_xlabel("artifact size (GiB, incl. bf16 vision tower)")
ax.set_ylabel("KL to bf16  (millinats/token, log scale)")
ax.set_title("Qwen3.6-35B-A3B — divergence vs size (lower is better)")
ax.grid(alpha=0.3, which="both")
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig("qwen36_ladder.png", dpi=200)
print("wrote qwen36_ladder.png")
