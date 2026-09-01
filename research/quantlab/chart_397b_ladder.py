#!/usr/bin/env python3
"""397B ladder chart: prose ppl vs size, real artifacts, same referee."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

flat = [(100.9, 3.1706, "flat K128"), (112.0, 2.7655, "flat K256"),
        (144.0, 2.3519, "flat K2048")]
harvest = [(97.2, 3.2730, "harvest, 97 GiB", (-4, 12)), (99.05, 3.2289, "harvest, 99 GiB", (-92, -4)),
           (107.9, 2.7790, "harvest, 108 GiB", (-92, -4))]
spicy = [(121.0, 3.1843, "spicyneuron 2.6bit"), (165.6, 2.3614, "spicyneuron 3.5bit")]

fig, ax = plt.subplots(figsize=(9, 5.6), dpi=150)
fx, fy = [p[0] for p in flat], [p[1] for p in flat]
ax.plot(fx, fy, "o-", color="#2563eb", lw=2, ms=8, label="ours — flat VQ ladder", zorder=4)
hx, hy = [p[0] for p in harvest], [p[1] for p in harvest]
ax.plot(hx, hy, "s", color="#0d9488", ms=8, label="ours — shallow-harvest (fills the gaps)", zorder=4)
sx, sy = [p[0] for p in spicy], [p[1] for p in spicy]
ax.plot(sx, sy, "D", color="#9ca3af", ms=8, label="spicyneuron (community, text-only)", zorder=3)

for x, y, t in flat:   ax.annotate(t, (x, y), textcoords="offset points", xytext=(10, 4), fontsize=8.5)
for x, y, t, off in harvest: ax.annotate(t, (x, y), textcoords="offset points", xytext=off, fontsize=8.5, color="#0f766e")
for x, y, t in spicy:  ax.annotate(t, (x, y), textcoords="offset points", xytext=(8, 6), fontsize=8.5, color="#6b7280")

# the interpolation-gap story
ax.plot([112.0, 144.0], [2.7655, 2.3519], "--", color="#93c5fd", lw=1.2, zorder=2)
ax.annotate("the 32-GiB gap between flat rungs —\nharvest fills it, beating interpolation",
            (113.5, 2.95), fontsize=8.5, color="#0f766e", ha="left")

ax.set_xlabel("size on disk (GiB)")
ax.set_ylabel("referee perplexity, prose (lower is better)")
ax.set_title("Qwen3.5-397B — quality vs size, all points measured on one instrument\n"
             "(ours keep vision; spicyneuron rungs are text-only)", fontsize=11)
ax.grid(alpha=0.25)
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig("chart_397b_ladder.png")
print("wrote chart_397b_ladder.png")
