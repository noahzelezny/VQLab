#!/usr/bin/env python3
"""Regenerate RESULTS-V5.md from the saved per-position KL arrays.

Every number in the tables is computed here from
vqlab-scratch/paper_rev/perpos_<family>{_v2}/<row>__<corpus>.safetensors and
its .json record. VQ rows are the v2-runtime scores (night 4); affine and
spicyneuron rows are the earlier scores (the VQ runtime profile does not touch
an affine artifact). A pair is formed only when both arrays were scored
against the SAME teacher cache (checked), so positions match.

    python research/paper/make_results.py > research/paper/RESULTS-V5.md
"""
import json
import math
import pathlib
import struct
import sys

import numpy as np
from safetensors.numpy import load_file

PR = pathlib.Path("<scratch>/paper_rev")
CORPORA = ("prose", "code", "lit")
PARAMS = {"397b": 396.35e9, "35b": 34.66e9, "27b": 26.90e9}

# row -> (per-pos dir, artifact dir for sizing, label)
V2 = lambda f: PR / f"perpos_{f}_v2"
OLD = lambda f: PR / f"perpos_{f}"
PIN, V = PR / "pin", PR / "pinv2"
ROWS = {
 "397b": [
  ("r22flat", V2, V / "q397_v1flat", "VQ flat d4/K128 (2.2 v1)"),
  ("floor_d4k128", V2, V / "r397p_floor", "VQ flat d4/K128, second fit"),
  ("r22v2", V2, V / "q397_2.2", "VQ-2.2bpw (v2 mix)"),
  ("r24", V2, V / "q397_2.4", "VQ-2.4bpw"),
  ("r26", V2, V / "q397_2.6", "VQ-2.6bpw"),
  ("spicy26", OLD, PIN / "spicyneuron--Qwen3.5-397B-A17B-MLX-2.6bit", "spicyneuron 2.6-bit"),
  ("r31", V2, V / "q397_3.1", "VQ-3.1bpw"),
  ("spicy35", OLD, None, "spicyneuron 3.5-bit"),
 ],
 "35b": [
  ("d2k16", V2, V / "r35p_d2k16", "VQ d2/K16"),
  ("e112_A", V2, V / "r35_e112_A", "VQ d4/K256 (unweighted)"),
  ("e112_B", V2, V / "r35_e112_B", "VQ d4/K256 (tail-weighted)"),
  ("r34", V2, V / "q35_3.4", "VQ-3.4bpw"),
  ("q3", OLD, None, "affine q3"),
  ("r38", V2, V / "q35_3.8", "VQ-3.8bpw"),
  ("d4k16384", V2, V / "r35p_d4k16384", "VQ d4/K16384"),
  ("d2k256", V2, V / "r35p_d2k256", "VQ d2/K256"),
  ("q4", OLD, None, "affine q4"),
  ("r54", V2, V / "q35_5.4", "VQ-5.4bpw (d2/K1024)"),
  ("floor_d2k1024", V2, V / "r35p_floor_d2k1024", "VQ d2/K1024, second fit"),
  ("d2k4096", V2, V / "r35p_d2k4096", "VQ d2/K4096"),
  ("q6", OLD, None, "affine q6"),
  ("q8", OLD, None, "affine q8"),
 ],
 "27b": [
  ("q2", OLD, None, "affine q2"),
  ("d4k256", V2, V / "r27_d4k256", "VQ d4/K256"),
  ("d4k1024", V2, V / "r27_d4k1024", "VQ d4/K1024"),
  ("q3", OLD, None, "affine q3"),
  ("d2k64", V2, V / "r27_d2k64", "VQ d2/K64"),
  ("r39", V2, V / "q27_3.9", "VQ-3.9bpw (d4/K4096)"),
  ("r45", V2, V / "q27_4.5", "VQ-4.5bpw (d2/K256)"),
  ("floor_d2k256", V2, V / "r27_d2k256_floor", "VQ d2/K256, second fit"),
  ("q4", OLD, None, "affine q4"),
  ("r48", V2, V / "q27_4.8", "VQ-4.8bpw"),
  ("d2k4096", V2, V / "r27_d2k4096", "VQ d2/K4096"),
  ("q6", OLD, None, "affine q6"),
  ("q8", OLD, None, "affine q8"),
 ],
}
# (family, arm, reference)  -- arm minus reference, negative = arm better
PAIRS = [
 ("397b", "r24", "spicy26"), ("397b", "r26", "spicy26"), ("397b", "r22flat", "spicy26"),
 ("397b", "r22v2", "spicy26"), ("397b", "r31", "spicy35"),
 ("35b", "r34", "q3"), ("35b", "r38", "q4"), ("35b", "d2k256", "q4"),
 ("35b", "r54", "q6"), ("35b", "d2k4096", "q6"),
 ("27b", "r39", "q3"), ("27b", "d4k1024", "q3"), ("27b", "r45", "q4"), ("27b", "r48", "q4"),
 ("27b", "d2k4096", "q6"),
]
FLOORS = [("27b", "r45", "floor_d2k256", "d2/K256"), ("35b", "r54", "floor_d2k1024", "d2/K1024"),
          ("397b", "r22flat", "floor_d4k128", "d4/K128")]
E112 = [("35b", "e112_B", "e112_A")]

# text-weight sizes for rows whose artifact is not on hand, from the prior table
KNOWN_GIB = {("397b", "spicy35"): 165.57, ("35b", "q3"): 14.14, ("35b", "q4"): 18.17,
             ("35b", "q6"): 26.23, ("35b", "q8"): 34.30, ("27b", "q2"): 7.83,
             ("27b", "q3"): 10.96, ("27b", "q4"): 14.09, ("27b", "q6"): 20.36, ("27b", "q8"): 26.62}
# Text weights = tensors under language_model.* -- the MTP sidecar (block.*, fc.*,
# norm_* in mtp-head-*.safetensors) and vision towers (vision_tower.*, model.*)
# are outside it. A name filter on "mtp" missed the sidecar by 5.41 GiB.
TEXT_PREFIX = "language_model."


def text_bytes(art: pathlib.Path) -> int:
    n = 0
    for f in art.glob("*.safetensors"):
        with open(f, "rb") as fh:
            hlen = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(hlen))
        for k, v in hdr.items():
            if not k.startswith(TEXT_PREFIX):
                continue
            a, b = v["data_offsets"]
            n += b - a
    return n


def rec(fam, row, where):
    d = where(fam)
    out = {}
    for c in CORPORA:
        p = d / f"{row}__{c}.safetensors"
        j = json.load(open(str(p) + ".json"))
        a = load_file(str(p))
        out[c] = {"kl": a["kl_millinats"].astype(np.float64), "json": j,
                  "cache": pathlib.Path(j.get("_cache_dir", "")).name}
    return out


def pair(a, b):
    d = a - b
    sem = d.std(ddof=1) / math.sqrt(len(d))
    return d.mean(), d.mean() / sem, (d < 0).mean()


def main():
    data, size = {}, {}
    for fam, rows in ROWS.items():
        for row, where, art, _ in rows:
            data[fam, row] = rec(fam, row, where)
            size[fam, row] = (text_bytes(art) / 2**30) if art is not None and art.exists() \
                else KNOWN_GIB[fam, row]
    w = sys.stdout.write
    w("# Paper v5 results — full-vocabulary KL, v2 runtime\n\n"
      "Generated by `research/paper/make_results.py` from the saved per-position arrays; "
      "do not edit by hand.\n\n"
      "All KL: exact KL(teacher‖student) over the full 248,320-token vocabulary, mnats, "
      "12,288 positions per corpus, paired on identical positions. VQ rows are scored on the "
      "v2 runtime (both bf16-I/O flags on); affine and spicyneuron rows are unaffected by the "
      "VQ runtime. Sizes: text weights (no vision tower, no MTP head); "
      "bpw = text bytes × 8 / text parameters. The 397B VQ rows are not yet generation-smoked "
      "on v2 (too large for one box; cluster smoke pending).\n")
    for fam, rows in ROWS.items():
        w(f"\n## {fam.upper()}  ({PARAMS[fam] / 1e9:.2f}B text parameters)\n\n"
          "| row | what | GiB | bpw | prose | code | literary | prose top-1 | prose median "
          "| code median | lit median |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
        for row, _, _, label in sorted(rows, key=lambda r: size[fam, r[0]]):
            r, g = data[fam, row], size[fam, row]
            bpw = g * 2**30 * 8 / PARAMS[fam]
            m = {c: r[c]["kl"].mean() for c in CORPORA}
            md = {c: float(np.median(r[c]["kl"])) for c in CORPORA}
            w(f"| {row} | {label} | {g:.2f} | {bpw:.2f} | {m['prose']:.1f} | {m['code']:.1f} | "
              f"{m['lit']:.1f} | {r['prose']['json']['top1_agreement'] * 100:.1f}% | "
              f"{md['prose']:.2f} | {md['code']:.2f} | {md['lit']:.2f} |\n")

    def pair_table(title, items, note=""):
        w(f"\n## {title}\n\n{note}| family | arm (GiB) | reference (GiB) | corpus | ref | arm "
          "| diff % | t | positions arm better |\n|---|---|---|---|---|---|---|---|---|\n")
        for fam, arm, ref, *_ in items:
            A, R = data[fam, arm], data[fam, ref]
            for c in CORPORA:
                if A[c]["cache"] != R[c]["cache"]:
                    raise SystemExit(f"{fam} {arm} vs {ref} {c}: different caches, cannot pair")
                dm, t, better = pair(A[c]["kl"], R[c]["kl"])
                rm = R[c]["kl"].mean()
                w(f"| {fam.upper()} | {arm} ({size[fam, arm]:.1f}) | {ref} ({size[fam, ref]:.1f}) | "
                  f"{c} | {rm:.1f} | {A[c]['kl'].mean():.1f} | {dm / rm * 100:+.1f}% | {t:+.1f} | "
                  f"{better * 100:.0f}% |\n")

    pair_table("Noise floors (independent second fit of the same geometry, paired)", FLOORS,
               "Second fits use the current plain fitter (k-means++, plain Lloyd, max-abs scales); "
               "the originals keep the fitter version they shipped with, so a floor bounds draw "
               "and fitter-version spread together.\n\n")
    pair_table("Paired comparisons the paper makes (arm − reference; negative = arm better)", PAIRS)
    pair_table("Tail weighting at fixed bytes (35B, d4/K256, plain fitter, seed 1234)", E112,
               "Arm = magnitude weighting p=4 on layers ≥13; reference = unweighted. "
               "Byte-identical sizes.\n\n")


if __name__ == "__main__":
    main()
