#!/usr/bin/env python3
"""Paper figures. Values are the LEDGER's current truth (paper/LEDGER.md).

Supersedes the repo-root chart_397b_ladder.py, whose spicyneuron point sat at
121.0 GiB (the record says 120.6) and which predates the K512, d8 and refit
rungs. Writes into paper/ only.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OURS, AFFINE = "#2563eb", "#9ca3af"

# ---- 397B: prose ppl vs packed post-graft GiB -------------------------------
flat = [(100.93, 3.1706, "flat K128"),
        (111.62, 2.7655, "flat K256"),
        (122.31, 2.5634, "flat K512"),
        (143.68, 2.3410, "flat K2048 (refit)")]
d8 = [(100.97, 3.0591, "d8/K16384")]
harvest = [(97.20, 3.2730, None), (99.05, 3.2289, None),
           (107.90, 2.7790, None), (139.93, 2.3452, None)]
spicy = [(120.57, 3.1843, "spicyneuron 2.6bit"), (165.57, 2.3614, "spicyneuron 3.5bit")]

fig, ax = plt.subplots(figsize=(8.4, 5.2))
ax.plot([p[0] for p in flat], [p[1] for p in flat], "o-", color=OURS, ms=7,
        lw=1.8, label="ours — flat VQ ladder", zorder=4)
ax.plot([p[0] for p in d8], [p[1] for p in d8], "s", color=OURS, ms=8,
        mfc="white", mew=2, label="ours — d8/K16384 (published)", zorder=5)
ax.plot([p[0] for p in harvest], [p[1] for p in harvest], "^", color=OURS,
        ms=6, alpha=.55, ls="none", label="ours — harvest rungs", zorder=3)
ax.plot([p[0] for p in spicy], [p[1] for p in spicy], "D", color=AFFINE, ms=8,
        ls="none", label="spicyneuron (hand-tuned mixed affine, text-only)", zorder=3)
OFFS = {"d8/K16384": (10, -12), "flat K128": (8, 4),
        "spicyneuron 3.5bit": (-118, 10)}
for x, y, t in flat + d8 + spicy:
    if t:
        ax.annotate(t, (x, y), textcoords="offset points",
                    xytext=OFFS.get(t, (7, 6)), fontsize=8.5, color="#374151")
ax.set_xlabel("packed size, post-graft (GiB)")
ax.set_ylabel("wikitext perplexity (lower is better)")
ax.set_title("Qwen3.5-397B-A17B — size vs quality, one instrument\n"
             "(ours keep the vision tower; spicyneuron rungs are text-only)",
             fontsize=11)
ax.grid(alpha=.25); ax.legend(fontsize=8.5, loc="upper right")
fig.tight_layout(); fig.savefig("paper/fig_397b_ladder.png", dpi=200)

# ---- 35B and dense 27B: KL vs packed GiB, log y -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))

# 35B sizes are MEASURED POST-GRAFT: every rung carries the 0.832 GiB bf16
# vision tower, as the community comparators always did. Our q6 was grafted
# 08-24; q4 and q8 are community builds that already had it.
vq35 = [(15.670, 53.022, "d4/K8192"), (16.615, 47.535, "d4/K16384"),
        (18.475, 36.862, None), (22.226, 28.141, "d2/K1024"),
        (25.977, 25.502, "d2/K4096")]
aff35 = [(19.000, 78.557, "4-bit"), (27.066, 13.358, "6-bit"),
         (35.131, 7.449, "8-bit")]
# 27B sizes are MEASURED POST-GRAFT (+0.858 GiB vision tower on rungs AND
# comparators alike, 08-24). d2/K512 is E142 arm 2, the published artifact.
vq27 = [(10.47, 325.6, None), (11.47, 148.5, "d4/K1024"), (12.47, 85.8, None),
        (14.45, 40.3, "d2/K256"), (15.45, 32.8, "d2/K512"), (18.44, 26.7, None)]
aff27 = [(8.69, 1426.9, None), (11.82, 187.8, "q3"), (14.95, 45.8, "q4"),
         (21.21, 3.71, "q6"), (27.48, 1.254, "q8")]  # q8 REBUILT (E144)

for ax, vq, aff, title in [
        (axes[0], vq35, aff35, "Qwen3.6-35B-A3B (MoE)"),
        (axes[1], vq27, aff27, "Qwen3.8-27B (dense)")]:
    ax.plot([p[0] for p in vq], [p[1] for p in vq], "o-", color=OURS, ms=6,
            lw=1.8, label="ours (data-free VQ)")
    ax.plot([p[0] for p in aff], [p[1] for p in aff], "D--", color=AFFINE,
            ms=6, lw=1.4, label="affine")
    OFF2 = {"q3": (6, 7), "d4/K1024": (-58, -3), "q4": (7, 6),
            "d2/K256": (-52, -10), "d2/K512": (6, -12),
            "d4/K16384": (8, -11), "8-bit": (-30, 8), "4-bit": (6, 4)}
    for x, y, t in vq + aff:
        if t:
            ax.annotate(t, (x, y), textcoords="offset points",
                        xytext=OFF2.get(t, (6, 5)), fontsize=8,
                        color="#374151")
    ax.set_yscale("log"); ax.set_xlabel("packed size (GiB)")
    ax.set_title(title, fontsize=11); ax.grid(alpha=.25, which="both")
axes[0].set_ylabel("KL to bf16 (mnats, log scale — lower is better)")
axes[0].legend(fontsize=8.5)
fig.tight_layout(); fig.savefig("paper/fig_35b_27b.png", dpi=200)
print("wrote paper/fig_397b_ladder.png and paper/fig_35b_27b.png")
