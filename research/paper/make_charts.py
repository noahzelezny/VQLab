#!/usr/bin/env python3
"""Paper v5 figures, one panel per corpus, from the same per-position arrays
and sizes as RESULTS-V5.md (make_results.py). No hand-typed values.

    python research/paper/make_charts.py

Writes fig_397b_ladder.png and fig_35b_27b.png beside this file.
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import make_results as mr

HERE = pathlib.Path(__file__).resolve().parent
VQ, AFF, PUB = "#1d4ed8", "#b45309", "#9ca3af"
CORPUS_NAME = {"prose": "prose", "code": "code", "lit": "literary"}

# (family, rows, style, label) -- rows in size order are joined by a line
SERIES = {
 "397b": [
  (["r24_sk8", "r26_sk8", "r31_sk8"], "o-", VQ, "VQ experts, spicyneuron skeleton"),
  (["spicy26", "spicy35"], "D", AFF, "spicyneuron affine (same skeleton)"),
  (["r22flat", "r24", "r26", "r31"], "o--", PUB, "VQ as published (6-bit skeleton)"),
 ],
 "35b": [
  (["d2k16", "r34", "r38", "d4k16384", "d2k256", "r54", "d2k4096"], "o-", VQ,
   "VQ experts"),
  (["s2", "s3", "s4", "s5", "s6", "s8"], "D-", AFF, "affine experts (same skeleton)"),
  (["q3", "q4", "q6", "q8"], "s--", PUB, "uniform affine as published"),
 ],
 "27b": [
  (["d4k256", "d4k1024", "r39", "r45", "r48", "d2k4096"], "o-", VQ, "VQ MLPs"),
  (["s2", "s3", "s4", "s5", "s6", "s8"], "D-", AFF, "affine MLPs (same skeleton)"),
  (["q2", "q3", "q6", "q8"], "s--", PUB, "uniform affine as published"),
 ],
}
TITLE = {"397b": "Qwen3.5-397B-A17B", "35b": "Qwen3.6-35B-A3B (MoE)",
         "27b": "Qwen3.8-27B (dense)"}


def load(fam):
    where = {row: (w, art) for row, w, art, _ in mr.ROWS[fam]}
    out = {}
    for rows, *_ in SERIES[fam]:
        for row in rows:
            w, art = where[row]
            r = mr.rec(fam, row, w)
            g = mr.TEXT_BYTES[fam, row] / 2**30 if (fam, row) in mr.TEXT_BYTES \
                else mr.KNOWN_GIB[fam, row]
            out[row] = (g, {c: r[c]["kl"].mean() for c in mr.CORPORA})
    return out


def panel_row(axes, fam, legend):
    d = load(fam)
    for ax, c in zip(axes, mr.CORPORA):
        for rows, style, color, label in SERIES[fam]:
            pts = sorted((d[r][0], d[r][1][c]) for r in rows)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], style, color=color,
                    ms=5, lw=1.5, mfc="white" if color == PUB else color,
                    label=label, zorder=2 if color == PUB else 3)
        ax.set_yscale("log")
        ax.set_title(f"{TITLE[fam]} — {CORPUS_NAME[c]}", fontsize=9.5)
        ax.grid(alpha=.25, which="both")
        ax.set_xlabel("text weights (GiB)", fontsize=8.5)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("KL to bf16 (mnats, log)", fontsize=8.5)
    if legend:
        if fam == "397b":   # every corner of these panels holds data: legend below
            h, l = axes[0].get_legend_handles_labels()
            axes[0].figure.legend(h, l, fontsize=8, loc="lower center", ncol=3, frameon=False)
        else:
            axes[0].legend(fontsize=7.5, loc="lower left")


def main():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    panel_row(axes, "397b", True)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(HERE / "fig_397b_ladder.png", dpi=200)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.4))
    panel_row(axes[0], "35b", True)
    panel_row(axes[1], "27b", True)
    fig.tight_layout()
    fig.savefig(HERE / "fig_35b_27b.png", dpi=200)
    print(f"wrote {HERE / 'fig_397b_ladder.png'} and {HERE / 'fig_35b_27b.png'}")


if __name__ == "__main__":
    main()
