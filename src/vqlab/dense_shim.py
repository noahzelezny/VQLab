"""The dense-artifact loader shim, as a standalone data file.

This text is concatenated after the dense runtimes (vq_switch.py +
vq_dense.py) to form a dense artifact's self-contained `model.py`. Stock
mlx-lm imports Model/ModelArgs from that file, so a downloader needs no
local code.

It lives here as its own module because the historical extraction path —
string-splitting a SHIM literal out of a gemma-specific build script —
broke the moment that script was refactored, and a shim-less model.py
fails at load with an opaque error.
"""

SHIM = '''


# ---------------------------------------------------------------------------
# mlx_lm `model_file` shim. Stock mlx_lm imports Model/ModelArgs from THIS
# file (it lives inside the checkpoint), so a user needs no local code. We
# reuse the registry architecture and swap the VQ'd modules for their dense
# VQ drop-ins BEFORE weights load, so shapes match at load time.
# ---------------------------------------------------------------------------
import importlib as _importlib
import json as _json
import pathlib as _pathlib

_cfg = _json.load(open(_pathlib.Path(__file__).parent / "config.json"))
_arch = _importlib.import_module(f"mlx_lm.models.{_cfg['model_type']}")
ModelArgs = _arch.ModelArgs


def _reach(root, path):
    obj = root
    parts = path.split(".")
    for c in parts[:-1]:
        obj = obj[int(c)] if c.isdigit() else getattr(obj, c)
    return obj, parts[-1]


class Model(_arch.Model):
    def __init__(self, args):
        super().__init__(args)
        for _p, _m in _cfg.get("vq_linear", {}).items():
            _obj, _leaf = _reach(self, _p)
            _pb = _m.get("pack_bits", 0)
            _ct = mx.uint32 if _pb else (mx.uint8 if _m["k"] <= 256 else mx.uint16)
            _cols = (_m["in"] // _m["dim"] // 32 * _pb) if _pb else _m["in"] // _m["dim"]
            setattr(_obj, _leaf, VQLinear(
                mx.zeros((_m["out"], _cols), dtype=_ct),
                mx.zeros((_m["k"], _m["dim"]), dtype=mx.float16),
                mx.zeros((_m["out"], _m["in"] // _m["group"]),
                         dtype=mx.float16),
                group_size=_m["group"], pack_bits=_pb,
                in_features=_m["in"] if _pb else None))
        for _p, _m in _cfg.get("vq_embed", {}).items():
            _obj, _leaf = _reach(self, _p)
            _pb = _m.get("pack_bits", 0)
            _ct = mx.uint32 if _pb else (mx.uint8 if _m["k"] <= 256 else mx.uint16)
            _cols = (_m["in"] // _m["dim"] // 32 * _pb) if _pb else _m["in"] // _m["dim"]
            setattr(_obj, _leaf, VQEmbedding(
                mx.zeros((_m["rows"], _cols), dtype=_ct),
                mx.zeros((_m["k"], _m["dim"]), dtype=mx.float16),
                mx.zeros((_m["rows"], _m["in"] // _m["group"]),
                         dtype=mx.float16),
                group_size=_m["group"], pack_bits=_pb,
                in_features=_m["in"] if _pb else None))
'''
