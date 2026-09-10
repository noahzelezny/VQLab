#!/usr/bin/env python3
"""Regenerate chart_e4b_vqple.png for the gemma-4-e4b-it-VQ-PLE card.

No generator was kept for the original, which is how it drifted: it labelled
the incumbent 8.48 GiB when every measurement says 8.38 (EXPERIMENTS.md
E-series; `du` on the artifact reads 8.3G). It also stacked the VQ-PLE and
8-bit labels on top of each other — the two points sit ~1 GiB and ~0.7 mnats
apart, so a single shared offset made both unreadable.

Labels are placed per point here rather than by a shared rule, because with
four points that is the thing that actually keeps it legible.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#      size GiB,  KL mnats, label,               colour,    (dx, dy), ha
PTS = [
    (6.340, 20.830, "full VQ (rejected)",  "#d62728", (16,   4), "left"),
    (7.390,  7.451, "VQ-PLE (this work)",  "#2ca02c", (-12, -30), "center"),
    (8.380,  8.149, "8-bit incumbent",     "#1f77b4", (14,   14), "left"),
    (14.790, 0.000, "bf16 (teacher)",      "#7f7f7f", (0,    16), "center"),
]

fig, ax = plt.subplots(figsize=(8.0, 4.8))

for x, y, name, colour, (dx, dy), ha in PTS:
    hero = "this work" in name
    ax.scatter([x], [y], s=190 if hero else 130, color=colour, zorder=3,
               edgecolor="white", linewidth=1.4 if hero else 0)
    size = f"{x:.2f} GiB" if x != 14.79 else f"{x:.2f} GiB"
    detail = f"{size} · {y:.2f} mnats" if y else size
    ax.annotate(f"{name}\n{detail}", (x, y), textcoords="offset points",
                xytext=(dx, dy), ha=ha, fontsize=8.5, color=colour,
                fontweight="bold" if hero else "normal", linespacing=1.35,
                zorder=4)

# headroom so no label runs into an axis
ax.set_xlim(5.6, 16.6)
ax.set_ylim(-3.2, 25.5)

ax.set_xlabel("size on disk (GiB)")
ax.set_ylabel("KL to bf16  (millinats/token, lower = closer)")
ax.set_title("gemma-4-e4b: VQ-PLE is smaller AND closer to bf16 than the 8-bit")
ax.grid(alpha=0.3)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

fig.tight_layout()
fig.savefig("chart_e4b_vqple.png", dpi=200)
print("wrote chart_e4b_vqple.png")
