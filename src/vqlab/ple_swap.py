"""Swap an artifact's PLE tables for another rung's, byte-for-byte.

    vqlab ple-swap --artifact <base rung> --donor <other rung> --out <dir>

The PLE tables are ~23% of a Flash-Next artifact (17.6 GB on the 3.2 at 55
bytes/row; the 2.1 ships them at 20, the 4.4 at 80) and every rung fits
them at ONE geometry chosen by hand. This builds the arm that prices those
bytes on KL: the base rung's experts, attention and runtime, with the
donor's PLE shards and `vq_ple` config block.

Pure IO, zero fitting, zero copying: every PLE-bearing shard in this family
holds ONLY PLE tensors, so the output is symlinks (base shards + donor PLE
shards under their own names) plus a rewritten index and config.
Refuses if a PLE-bearing shard on either side carries any other tensor.
"""
import argparse
import json
import os
import pathlib
import shutil
import sys


def _ple_key(k):
    return ".ngram_embedding." in k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True, help="base rung")
    ap.add_argument("--donor", required=True, help="rung whose PLE bytes to take")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base, donor, out = (pathlib.Path(p).resolve() for p in (a.artifact, a.donor, a.out))

    def index(p):
        return json.load(open(p / "model.safetensors.index.json"))

    bi, di = index(base), index(donor)
    bwm, dwm = bi["weight_map"], di["weight_map"]
    for name, wm in (("base", bwm), ("donor", dwm)):
        ple_shards = {f for k, f in wm.items() if _ple_key(k)}
        riders = [k for k, f in wm.items() if f in ple_shards and not _ple_key(k)]
        if riders:
            sys.exit(f"REFUSED: {name} PLE shards carry non-PLE tensors "
                     f"({len(riders)}, e.g. {riders[0]}); a symlink swap "
                     f"would drop them")
    bkeys = {k for k in bwm if _ple_key(k)}
    dkeys = {k for k in dwm if _ple_key(k)}
    if bkeys != dkeys:
        sys.exit(f"REFUSED: PLE key sets differ (base {len(bkeys)}, donor "
                 f"{len(dkeys)}, symmetric diff {len(bkeys ^ dkeys)})")

    out.mkdir(parents=True, exist_ok=True)
    for f in base.iterdir():
        if f.name.endswith(".safetensors") or f.name == "__pycache__":
            continue
        dst = out / f.name
        if dst.is_dir():
            shutil.rmtree(dst)
        elif dst.exists() or dst.is_symlink():
            dst.unlink()
        if f.is_dir():
            shutil.copytree(f, dst)
        else:
            shutil.copy(f.resolve(), dst)

    wm = {}
    linked = 0
    for k, f in bwm.items():
        if _ple_key(k):
            continue
        dst = out / f
        if not dst.is_symlink():
            dst.symlink_to((base / f).resolve())
            linked += 1
        wm[k] = f
    # Loaders glob `model*.safetensors` (mlx-lm and the streamed scorer
    # both), so the donor links MUST keep a `model` prefix or they are
    # silently never read ("missing parameter" at load). The base's own PLE
    # shards are not linked, so the donor's names are free unless they
    # collide with a base non-PLE shard.
    base_used = set(wm.values())
    dshards = {}
    for k in dkeys:
        f = dwm[k]
        nf = f if f not in base_used else "model-donor-" + f
        if f not in dshards:
            dst = out / nf
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to((donor / f).resolve())
            dshards[f] = nf
            linked += 1
        wm[k] = nf
    bi["weight_map"] = wm
    bi["metadata"]["total_size"] = sum(
        (out / f).resolve().stat().st_size for f in set(wm.values()))
    json.dump(bi, open(out / "model.safetensors.index.json", "w"), indent=1)

    cfg = json.load(open(out / "config.json"))
    dcfg = json.load(open(donor / "config.json"))
    cfg["vq_ple"] = dcfg["vq_ple"]
    json.dump(cfg, open(out / "config.json", "w"), indent=1)
    g = dcfg["vq_ple"]["geometry"]
    print(f"ple-swap: {len(dkeys)} PLE keys from {donor.name} "
          f"(d{g['dim']}-K{g['k']}, {g.get('row_bytes')} B/row) onto "
          f"{base.name}; {linked} symlinks, "
          f"{bi['metadata']['total_size']/2**30:.2f} GiB -> {out}")


if __name__ == "__main__":
    main()
