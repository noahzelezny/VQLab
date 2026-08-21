#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""Recompute model.safetensors.index.json metadata.total_size from the shards.

WHY. pack_artifact.py copies the SOURCE index verbatim (shutil.copy2), so a
packed artifact ships the UNPACKED total_size. exo reads that field to size the
model, computed 197.12 GiB for a 143.70 GiB artifact, and refused placement
with "No cycles found with sufficient memory" — a 37% overstatement. Any
downloader reading the index would be misled the same way.

Reads the safetensors header of each shard (8-byte little-endian length +
JSON) and sums each tensor's byte extent. Does not load tensors.
"""
import argparse, json, pathlib, struct, sys

ap = argparse.ArgumentParser()
ap.add_argument("--artifact", required=True)
ap.add_argument("--apply", action="store_true", help="write; otherwise report only")
args = ap.parse_args()

ART = pathlib.Path(args.artifact)
idx_path = ART / "model.safetensors.index.json"
idx = json.load(open(idx_path))
shards = sorted(set(idx["weight_map"].values()))

total = 0
for sh in shards:
    with open(ART / sh, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    for name, meta in hdr.items():
        if name == "__metadata__":
            continue
        s, e = meta["data_offsets"]
        total += e - s

old = idx.get("metadata", {}).get("total_size")
print(f"  artifact : {ART.name}")
print(f"  declared : {old if old is not None else '(absent)'}"
      + (f"  = {old/2**30:.2f} GiB" if old else ""))
print(f"  actual   : {total}  = {total/2**30:.2f} GiB")
if old == total:
    print("  OK — already correct, nothing to do.")
    sys.exit(0)
if old:
    print(f"  overstated by {(old-total)/2**30:.2f} GiB ({100*old/total-100:.1f}%)")
if not args.apply:
    print("  (report only; pass --apply to write)")
    sys.exit(0)

bak = idx_path.with_suffix(".json.pre_total_size")
if not bak.exists():
    bak.write_text(idx_path.read_text())
idx.setdefault("metadata", {})["total_size"] = total
idx_path.write_text(json.dumps(idx, indent=1))
check = json.load(open(idx_path))
assert check["metadata"]["total_size"] == total
assert check["weight_map"] == idx["weight_map"], "weight_map must be untouched"
print(f"  WROTE. backup at {bak.name}")
