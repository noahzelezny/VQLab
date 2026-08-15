#!/usr/bin/env python
"""Retrofit self-contained `model_file` loading into an EXISTING VQ codes
artifact (the 35B was built before the packaging existed). Writes model.py
(= quantlab/vq_switch.py + the loader shim, same text vq_397b_codes.py
generates) and adds config.json: model_file + vq_modules.

    ./add_model_file.py --artifact <dir> [--k 256 --dim 4 --group 64]
"""
import argparse
import json
import pathlib

import mlx.core as mx

ap = argparse.ArgumentParser()
ap.add_argument("--artifact", required=True)
ap.add_argument("--k", type=int, default=256)
ap.add_argument("--dim", type=int, default=4)
ap.add_argument("--group", type=int, default=64)
args = ap.parse_args()

ART = pathlib.Path(args.artifact)
idx = json.load(open(ART / "model.safetensors.index.json"))["weight_map"]
cfg = json.load(open(ART / "config.json"))

vq_modules = {}
by_shard = {}
for k, sh in idx.items():
    if k.endswith(".codes"):
        by_shard.setdefault(sh, []).append(k[:-6])
for sh, mods in sorted(by_shard.items()):
    data = mx.load(str(ART / sh))
    for m in mods:
        codes = data[m + ".codes"]
        cb = data[m + ".codebook"]
        E, out_d, nsub = codes.shape
        vq_modules[m] = {"experts": E, "out": out_d,
                         "in": nsub * cb.shape[1], "k": cb.shape[0],
                         "dim": cb.shape[1], "group": args.group}
    del data

cfg["model_file"] = "model.py"
cfg["vq_modules"] = vq_modules
json.dump(cfg, open(ART / "config.json", "w"), indent=1)

runtime = (pathlib.Path(__file__).parent / "vq_switch.py").read_text()
shim = '''

# ---------------------------------------------------------------------------
# mlx_lm `model_file` shim: stock mlx_lm imports Model/ModelArgs from THIS
# file (it lives inside the checkpoint). We reuse the registry architecture
# and swap each VQ'd expert module for VQSwitchLinear before weights load.
# ---------------------------------------------------------------------------
import importlib as _importlib
import json as _json
import pathlib as _pathlib

_cfg = _json.load(open(_pathlib.Path(__file__).parent / "config.json"))
_arch = _importlib.import_module(f"mlx_lm.models.{_cfg['model_type']}")
ModelArgs = _arch.ModelArgs


class Model(_arch.Model):
    def __init__(self, args):
        super().__init__(args)
        for _path, _m in _cfg.get("vq_modules", {}).items():
            _obj = self
            _parts = _path.split(".")
            for _c in _parts[:-1]:
                _obj = _obj[int(_c)] if _c.isdigit() else getattr(_obj, _c)
            _ct = mx.uint8 if _m["k"] <= 256 else mx.uint16
            setattr(_obj, _parts[-1], VQSwitchLinear(
                mx.zeros((_m["experts"], _m["out"], _m["in"] // _m["dim"]),
                         dtype=_ct),
                mx.zeros((_m["k"], _m["dim"]), dtype=mx.float16),
                mx.zeros((_m["experts"], _m["out"], _m["in"] // _m["group"]),
                         dtype=mx.float16),
                group_size=_m["group"],
            ))
'''
(ART / "model.py").write_text(runtime + shim)
print(f"wrote model.py + config keys: {len(vq_modules)} vq modules -> {ART}")
