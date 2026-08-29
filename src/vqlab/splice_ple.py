"""Splice VQ PLE codes into the packed artifact.

Removes each ngram module's affine tensors (weight/scales/biases) from the
artifact, copies the PLE fit shards in ({key}.codes/.codebook/.vq_scales),
rewrites the index, and updates config: the per-path affine quantization
entries for those modules are DELETED (the loader must not construct
QuantizedEmbedding for weights that no longer exist) and a "vq_ple" block
records geometry + keys for the runtime patch to consume.
"""
import argparse
import json, pathlib, shutil, sys
import mlx.core as mx

# Splicing is pure IO: lazy loads must never evaluate on the GPU stream —
# over SMB that lands storage reads inside a Metal command buffer and the
# watchdog kills it (M4, d8/K4096 assembly, 2026-08-29; fourth instance of
# the class). The whole script runs on CPU.
mx.set_default_device(mx.cpu)

_ap = argparse.ArgumentParser()
_ap.add_argument("--artifact", required=True)
_ap.add_argument("--ple-fit", required=True)
_a = _ap.parse_args()
ART = pathlib.Path(_a.artifact); PLE = pathlib.Path(_a.ple_fit)
idx_p = ART / "model.safetensors.index.json"
idx = json.load(open(idx_p)); wm = idx["weight_map"]
man = json.load(open(PLE / "ple_manifest.json"))

ngram_mods = {k.rsplit(".weight", 1)[0].replace("model.language_model.", "model.")
              for k in man["tensors"]}
# artifact keys use the sanitized mlx layout (model.layers...); map by suffix
drop = [k for k in wm if any(k.startswith(m + ".") or k == m + ".weight"
        for m in ngram_mods) or ".ngram_embedding." in k]
shards = sorted({wm[k] for k in drop})
print(f"{len(drop)} affine tensors to remove across {len(shards)} shards")

for sh in shards:
    data = mx.load(str(ART / sh))
    keep = {k: v for k, v in data.items() if k not in drop}
    if keep:
        # mx.load is LAZY: writing over the file the kept arrays still read
        # from corrupts both. Write a temp file, then swap.
        tmp = ART / sh.replace(".safetensors", ".tmp.safetensors")
        mx.save_safetensors(str(tmp), keep, metadata={"format": "mlx"})
        tmp.replace(ART / sh)
    else:
        (ART / sh).unlink()
    print(f"  {sh}: kept {len(keep)}/{len(data)}")
for k in drop:
    del wm[k]

added = 0
for t, meta in man["tensors"].items():
    src_shard = meta["shard"]
    dst = ART / ("model-" + src_shard)
    if not dst.exists():
        shutil.copy(PLE / src_shard, dst)
    art_key = t.replace("model.language_model.", "model.")
    if art_key.endswith(".weight"):
        art_key = art_key[: -len(".weight")]   # module path, not tensor path
    for suf in (".codes", ".codebook", ".vq_scales"):
        wm[art_key + suf] = "model-" + src_shard
        added += 1
# the fit shards carry keys under the SOURCE naming; rewrite them to match
for src_shard in sorted({m["shard"] for m in man["tensors"].values()}):
    data = mx.load(str(ART / ("model-" + src_shard)))
    fixed = {k.replace("model.language_model.", "model.")
              .replace(".weight.codes", ".codes")
              .replace(".weight.codebook", ".codebook")
              .replace(".weight.vq_scales", ".vq_scales"): v
             for k, v in data.items()}
    if list(fixed) != list(data):
        tmp = ART / ("model-" + src_shard).replace(".safetensors", ".tmp.safetensors")
        mx.save_safetensors(str(tmp), fixed, metadata={"format": "mlx"})
        tmp.replace(ART / ("model-" + src_shard))
print(f"added {added} vq tensors from {len(set(m['shard'] for m in man['tensors'].values()))} ple shards")

idx["metadata"]["total_size"] = sum((ART / f).stat().st_size
                                    for f in set(wm.values()))
json.dump(idx, open(idx_p, "w"), indent=1)

cfg_p = ART / "config.json"; cfg = json.load(open(cfg_p))
q = cfg.get("quantization", {})
removed = [m for m in list(q) if ".ngram_embedding." in m]
for m in removed:
    del q[m]
qc = cfg.get("quantization_config", {})
for m in removed:
    qc.pop(m, None)
cfg["vq_ple"] = {"geometry": man["geometry"],
                 "keys": sorted(k.replace("model.language_model.", "model.")
                                 .removesuffix(".weight")
                                for k in man["tensors"]),
                 "shapes": {k.replace("model.language_model.", "model.")
                             .removesuffix(".weight"): v["shape"]
                            for k, v in man["tensors"].items()}}
json.dump(cfg, open(cfg_p, "w"), indent=1)
print(f"config: removed {len(removed)} affine entries, added vq_ple "
      f"({len(cfg['vq_ple']['keys'])} keys)")
tot = sum((ART / f).stat().st_size for f in set(wm.values()))
print(f"artifact now {tot/2**30:.1f} GiB")
