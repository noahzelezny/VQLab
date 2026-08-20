#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""Assemble a RUNNABLE VQ e4b from the 8-bit incumbent + our VQ fits.

Start from mlx-community's 8-bit artifact (everything already quantized),
DROP the 8-bit tensors for the mlp trio and the PLE table, and splice in
the VQ codes/codebook/scales for those. Everything else — attention,
towers, norms, embed_tokens — stays exactly as the incumbent ships it, so
any measured difference is attributable to the VQ swap alone.

Writes model.py (vq_dense.py + a loader shim) so `mlx_lm` can load it with
no code on the user's machine, same self-contained pattern as the 397B and
26b artifacts.
"""
import argparse
import json
import pathlib
import shutil

import mlx.core as mx

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True, help="8-bit incumbent artifact")
ap.add_argument("--mlp", required=True, help="fit_e4b_vq.py output dir")
ap.add_argument("--ple", default=None, help="fit_e4b_ple.py output dir")
ap.add_argument("--out", required=True)
args = ap.parse_args()

BASE, OUT = pathlib.Path(args.base), pathlib.Path(args.out)
MLP = pathlib.Path(args.mlp)
OUT.mkdir(parents=True, exist_ok=True)

base_idx = json.load(open(BASE / "model.safetensors.index.json"))["weight_map"]
mlp_cfg = json.load(open(MLP / "config.json"))["vq_modules"]
mlp_w = mx.load(str(MLP / "model-00001-of-00001.safetensors"))

DROP_MLP = tuple(f".mlp.{p}." for p in ("gate_proj", "up_proj", "down_proj"))
PLE_KEY = "language_model.model.embed_tokens_per_layer"


# mlx-community's e4b-8bit ships k/v/k_norm tensors for the KV-SHARED
# layers, but mlx_lm's gemma4 does not instantiate those modules (has_kv is
# False for layer_idx >= num_hidden_layers - num_kv_shared_layers), so a
# strict load of the incumbent FAILS with "126 parameters not in model" and
# every consumer has been quietly falling back to strict=False all evening.
# Our artifact drops them: it then loads strictly and cleanly, and the bytes
# were dead weight for this runtime anyway. Recorded because a silent
# strict=False fallback is exactly the kind of thing that hides a real
# loading bug later.
_cfg_base = json.load(open(BASE / "config.json"))
_t = _cfg_base.get("text_config", _cfg_base)
FIRST_SHARED = _t["num_hidden_layers"] - _t.get("num_kv_shared_layers", 0)
DEAD_KV = (".self_attn.k_proj.", ".self_attn.v_proj.", ".self_attn.k_norm.")


def is_dropped(k):
    if k.startswith("language_model.model.layers.") and any(d in k for d in DROP_MLP):
        return True
    if args.ple and k.startswith(PLE_KEY):
        return True
    if k.startswith("language_model.model.layers.") and any(d in k for d in DEAD_KV):
        li = int(k.split("layers.")[1].split(".")[0])
        if li >= FIRST_SHARED > 0:
            return True
    return False


# ---- copy through everything we are NOT replacing, shard by shard
new_map, shard_no = {}, 0
carried = 0
for sh in sorted(set(base_idx.values())):
    data = mx.load(str(BASE / sh))
    keep = {k: v for k, v in data.items() if not is_dropped(k)}
    carried += len(keep)
    if not keep:
        continue
    shard_no += 1
    name = f"model-{shard_no:05d}.safetensors"
    mx.save_safetensors(str(OUT / name), keep)
    for k in keep:
        new_map[k] = name
    del data, keep
    mx.clear_cache()
print(f"carried {carried} tensors from the 8-bit base")

# ---- splice VQ mlp
# The fitter writes codes/scales as [1, OUT, *] so verify_artifact (which
# speaks the expert format) can check them. A DENSE module wants 2D, and
# 2D also keeps the venv's expert-shaped VQ hook off them — squeeze here.
mlp_2d = {}
for k, v in mlp_w.items():
    mlp_2d[k] = v[0] if (k.endswith((".codes", ".vq_scales")) and v.ndim == 3) else v
shard_no += 1
name = f"model-{shard_no:05d}.safetensors"
mx.save_safetensors(str(OUT / name), mlp_2d)
for k in mlp_2d:
    new_map[k] = name
print(f"spliced {len(mlp_2d)} VQ mlp tensors (codes squeezed to 2D)")

vq_linear = {m: dict(v) for m, v in mlp_cfg.items()}
vq_embed = {}

# ---- splice VQ PLE
if args.ple:
    ple = mx.load(str(pathlib.Path(args.ple) / "ple.safetensors"))
    shard_no += 1
    name = f"model-{shard_no:05d}.safetensors"
    out_t = {f"{PLE_KEY}.{k}": v for k, v in ple.items()}
    mx.save_safetensors(str(OUT / name), out_t)
    for k in out_t:
        new_map[k] = name
    K, D = ple["codebook"].shape
    vq_embed[PLE_KEY] = {"rows": int(ple["codes"].shape[0]),
                         "in": int(ple["codes"].shape[1] * D),
                         "k": int(K), "dim": int(D), "group": 64}
    print(f"spliced VQ PLE: {vq_embed[PLE_KEY]}")

json.dump({"metadata": {}, "weight_map": new_map},
          open(OUT / "model.safetensors.index.json", "w"), indent=1)

# ---- config + model.py
cfg = json.load(open(BASE / "config.json"))
cfg["vq_linear"] = vq_linear
cfg["vq_embed"] = vq_embed
cfg["model_file"] = "model.py"
json.dump(cfg, open(OUT / "config.json", "w"), indent=1)

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
(OUT / "model.py").write_text(
    pathlib.Path("vq_dense.py").read_text() + SHIM)

for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
          "generation_config.json", "processor_config.json", "preprocessor_config.json"):
    if (BASE / f).exists():
        shutil.copy(BASE / f, OUT / f)

tot = sum(p.stat().st_size for p in OUT.glob("*.safetensors"))
print(f"\nartifact -> {OUT}\n  {tot/2**30:.2f} GiB of tensors")
