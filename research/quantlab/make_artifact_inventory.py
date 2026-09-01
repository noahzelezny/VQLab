#!/usr/bin/env python3
"""Inventory every model artifact on the SSD -> ARTIFACTS.md + artifacts.json.

Read-only. Deletes nothing, hashes nothing by default. The point is to make a
few MB of record stand in for a few TB of weights, so that deleting the
weights costs no evidence.

    ./make_artifact_inventory.py                # inventory only
    ./make_artifact_inventory.py --hash KEEP    # + full sha256 for keepers

WHY FULL sha256 AND NOT A HEAD HASH: e95-27b-dense-vq and -vq-r2 have equal
byte counts, equal shard counts, and IDENTICAL head hashes on every shard.
They differ only in the full contents of shard 4 -- and one of them is the
artifact scored in the paper. A head hash cannot tell them apart. Anything
we intend to KEEP gets a real hash or it has no identity.

REBUILDABILITY is the axis that decides deletion, not size:
  * affine  -- mlx_lm.convert is deterministic. Re-running reproduces the
               bytes. Deleting costs nothing but time.
  * graft   -- the vision graft shard is byte-identical across every artifact
               in a family (verified 08-25: 921,497,299 B, one sha256, three
               independently grafted 27B rungs). Reproducible.
  * packed  -- packing is a pure representation change, verified to 4 decimals
               against the unpacked twin. Reproducible from the twin.
  * vq-fit  -- UNSEEDED k-means. Re-running yields a DIFFERENT draw inside the
               noise floor. NOT reproducible. Deleting converts a measured
               number into a recorded one.
  * seeded  -- E142-27B arm 2 only. Reproducible bit-for-bit from recipe+seed.

Project ruling (Noah, 2026-08-25): a result that is fully written down is
enough. An artifact whose numbers live in EXPERIMENTS.md/LEDGER.md may be
deleted even when the fit is unseeded -- the write-up is the evidence, not the
bytes. This tool therefore reports rebuildability as INFORMATION, and does not
treat "unseeded" as a veto on deletion.
"""
import argparse, hashlib, json, os, subprocess, sys

ROOT = "/Volumes/Thunderbay SSD/Exo Models"


def du_gib(p):
    return int(subprocess.run(["du", "-sk", p], capture_output=True, text=True)
               .stdout.split()[0]) / 1048576.0


def read_index(d):
    """Return (total_bytes, n_tensors, n_vision, shards) from the index if present."""
    p = os.path.join(d, "model.safetensors.index.json")
    if not os.path.exists(p):
        return None
    try:
        j = json.load(open(p))
    except Exception:
        return None
    w = j.get("weight_map", {})
    shards = sorted(set(w.values()))
    tot = 0
    for f in shards:
        fp = os.path.join(d, f)
        if os.path.exists(fp):
            tot += os.path.getsize(fp)
    vis = sum(1 for k in w if k.startswith(("vision_tower", "model.visual", "embed_vision")))
    return tot, len(w), vis, shards


def classify(name, idx):
    n = name.lower()
    if n.startswith(("spicyneuron--", "mlx-community--", "youssofal--")):
        return "third-party", "redownload"
    if "bf16" in n or n.startswith(("qwen--", "google--")):
        return "base/teacher", "redownload"
    if n.startswith("thedrainflorist--"):
        return "ours/published", "on-hf"
    if n == "hub" or n.startswith("hub-"):
        return "cache", "redownload"
    if any(t in n for t in ("q2", "q3", "q4", "q6", "q8", "uniform-q", "affine")):
        return "ours/affine", "affine"
    if "packed" in n:
        return "ours/packed", "packed"
    return "ours/vq-fit", "vq-fit"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hash", choices=["KEEP", "ALL", "NONE"], default="NONE")
    ap.add_argument("--root", default=ROOT)
    a = ap.parse_args()
    if not os.path.isdir(a.root):
        sys.exit(f"not mounted: {a.root}")

    # Walk one level down into CONTAINER dirs -- a dir with no index of its own
    # that holds subdirs which do. The three *-rungs dirs are containers holding
    # 84 artifacts between them; treating each as a single blob hid every affine
    # comparator in the corpus and made the vq-fit bucket look enormous.
    targets = []
    for name in sorted(os.listdir(a.root)):
        d = os.path.join(a.root, name)
        if not os.path.isdir(d) or name.startswith("."):
            continue
        if read_index(d) is None:
            kids = [k for k in sorted(os.listdir(d))
                    if os.path.isdir(os.path.join(d, k)) and read_index(os.path.join(d, k))]
            if kids:
                targets += [(f"{name}/{k}", os.path.join(d, k)) for k in kids]
                continue
        targets.append((name, d))

    rows = []
    for name, d in targets:
        idx = read_index(d)
        cat, rebuild = classify(name, idx)
        rows.append({
            "name": name,
            "du_gib": round(du_gib(d), 3),
            "index_bytes": idx[0] if idx else None,
            "index_gib": round(idx[0] / 1073741824, 4) if idx else None,
            "tensors": idx[1] if idx else None,
            "vision_tensors": idx[2] if idx else None,
            "shards": idx[3] if idx else None,
            "category": cat,
            "rebuildable": rebuild,
            "sha256": None,
        })

    if a.hash != "NONE":
        want = [r for r in rows if a.hash == "ALL" or r["category"].startswith("ours")]
        for r in want:
            if not r["shards"]:
                continue
            h = {}
            for f in r["shards"]:
                fp = os.path.join(a.root, *r["name"].split("/"), f)
                if not os.path.exists(fp):
                    continue
                s = hashlib.sha256()
                with open(fp, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 24), b""):
                        s.update(chunk)
                h[f] = s.hexdigest()
            r["sha256"] = h
            print("hashed", r["name"], file=sys.stderr)

    json.dump(rows, open("artifacts.json", "w"), indent=1)

    by = {}
    for r in rows:
        by.setdefault(r["category"], []).append(r)
    out = ["# Artifact inventory", "",
           f"Generated from `{a.root}`. Sizes are `du` (block usage) and, where an",
           "index exists, the exact sum of the shard bytes it names. **`du` is not a",
           "size** -- an audit once mis-reported a 15.670 GiB artifact as 18.483 by",
           "reading blocks. Cite `index_gib`.", "",
           "`rebuildable` says what deleting costs. See make_artifact_inventory.py",
           "for what each value means; `vq-fit` is the only one that is not",
           "reproducible, and per the 08-25 ruling that is not a veto on deletion",
           "when the result is written down.", ""]
    tot = 0.0
    for cat in sorted(by):
        g = sum(r["du_gib"] for r in by[cat]); tot += g
        out.append(f"## {cat} — {g:,.1f} GiB, {len(by[cat])} artifacts\n")
        out.append("| artifact | du GiB | index GiB | tensors | vision | rebuildable |")
        out.append("|---|---|---|---|---|---|")
        for r in sorted(by[cat], key=lambda x: -x["du_gib"]):
            out.append("| `{name}` | {du_gib:,.1f} | {ig} | {t} | {v} | {rebuildable} |".format(
                ig=f"{r['index_gib']:,.4f}" if r["index_gib"] else "—",
                t=r["tensors"] or "—", v=r["vision_tensors"] or "—", **r))
        out.append("")
    out.insert(6, f"**Total: {tot:,.1f} GiB across {len(rows)} artifacts.**\n")
    open("ARTIFACTS.md", "w").write("\n".join(out))
    print(f"wrote ARTIFACTS.md and artifacts.json — {len(rows)} artifacts, {tot:,.1f} GiB")


if __name__ == "__main__":
    main()
