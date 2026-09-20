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

from vqlab.arch_resolve import ARRAYISH as _ARCH_ARRAYISH
from vqlab.arch_resolve import COERCE as _ARCH_COERCE
from vqlab.arch_resolve import PATHWALK as _ARCH_PATHWALK
from vqlab.arch_resolve import PRELUDE as _ARCH_PRELUDE
from vqlab.arch_resolve import VLM_SANITIZE as _ARCH_VLM_SANITIZE

ap = argparse.ArgumentParser()
ap.add_argument("--artifact", required=True)
ap.add_argument("--k", type=int, default=256)
ap.add_argument("--dim", type=int, default=4)
ap.add_argument("--group", type=int, default=64)
ap.add_argument("--adopt", default="", metavar="FLAG,FLAG|all",
               help="take the REPO's current default for these VQ_* flags "
                    "instead of preserving what the artifact shipped with. "
                    "By default a rebundle preserves every shipped default "
                    "and prints any the repo would have changed: rebundling "
                    "is normally done for an unrelated reason (a shim fix), "
                    "and silently adopting whatever defaults an arc has "
                    "landed since is how F148's VQ_DENSE_SS flip nearly rode "
                    "out on 17 artifacts it was never measured on. 'all' "
                    "restores the old take-everything behaviour.")
ap.add_argument("--runtime", choices=["v1.5", "v2"], default=None,
                help="runtime PROFILE to bake into the bundle. v1.5 (default) "
                     "leaves both bf16-I/O flags OFF: bit-exact with the arc6 "
                     "runtime every published artifact shipped, +8.1%% prefill "
                     "/ +12.2%% decode from the bit-exact gemmseg+walker work. "
                     "v2 turns them ON: +11.0%%/+19.2%%, at a small "
                     "family-local accuracy cost (F103/F105). An artifact "
                     "whose own quality gain pays for that cost may ship v2; "
                     "see docs/RUNTIME-SHIP-PLAN.md. DEFAULT IS NOW None: an "
                     "existing bundle keeps the profile it shipped with, "
                     "because defaulting to v1.5 SILENTLY DOWNGRADED a v2 "
                     "artifact on 2026-09-19. Pass it to force a profile.")
args = ap.parse_args()

ART = pathlib.Path(args.artifact)
idx = json.load(open(ART / "model.safetensors.index.json"))["weight_map"]
cfg = json.load(open(ART / "config.json"))

# DENSE REFUSAL (2026-09-07). This is the MoE bundler: it scans for 3-D
# `.codes` (expert modules) and splices vq_switch.py + the MoE shim. Run
# on a DENSE artifact it found no expert modules, wrote an empty
# vq_modules, and silently replaced a correct dense bundle (which must
# also carry vq_dense.py) with one that cannot serve — caught only
# because check-bundle then failed on three published 27B rungs. A
# bundler that quietly produces an unusable artifact is the exact class
# of defect the release gates exist for; refuse instead.
if cfg.get("vq_linear") or cfg.get("vq_embed"):
    raise SystemExit(
        "REFUSING: this is a DENSE artifact (config carries vq_linear/"
        "vq_embed). Its bundle must contain vq_switch.py AND vq_dense.py "
        "plus the dense shim — use build-dense, not bundle. Running this "
        "command would overwrite the dense runtime with a MoE-only one.")

vq_modules = {}
by_shard = {}
for k, sh in idx.items():
    if k.endswith(".codes"):
        by_shard.setdefault(sh, []).append(k[:-6])
prev = json.load(open(ART / "config.json")).get("vq_modules", {})
for sh, mods in sorted(by_shard.items()):
    data = mx.load(str(ART / sh))
    for m in mods:
        codes = data[m + ".codes"]
        cb = data[m + ".codebook"]
        if codes.ndim == 2:
            # PLE/embedding table (registered under config vq_ple) — not an
            # expert module.
            continue
        E, out_d, ncol = codes.shape
        # PACKED codes are uint32 words, so the last axis is WPR, not NSUB —
        # the shape no longer implies `in`. Bits come from the packer (via the
        # existing config); refuse to guess, because a wrong width decodes to
        # plausible-looking garbage rather than an error.
        if codes.dtype == mx.uint32:
            was = prev.get(m, {})
            bits = was.get("pack_bits")
            in_d = was.get("in")
            if not bits or not in_d:
                raise SystemExit(
                    f"{m}: codes are uint32 (packed) but config carries no "
                    "pack_bits/in for them. Run pack_artifact.py, which writes "
                    "both — do not retrofit a packed artifact by hand.")
        else:
            bits, in_d = 0, ncol * cb.shape[1]
        vq_modules[m] = {"experts": E, "out": out_d,
                         "in": in_d, "k": cb.shape[0],
                         "dim": cb.shape[1], "group": args.group}
        if bits:
            vq_modules[m]["pack_bits"] = bits
    del data

cfg["model_file"] = "model.py"
cfg["vq_modules"] = vq_modules
json.dump(cfg, open(ART / "config.json", "w"), indent=1)

runtime = (pathlib.Path(__file__).parent / "vq_switch.py").read_text()
import vqlab.runtime_profile as _ba
_shipped = (ART / "model.py").read_text() if (ART / "model.py").exists() else None
runtime, _report = _ba.resolve_runtime(
    _shipped, runtime, profile=args.runtime,
    adopt=tuple(f for f in args.adopt.split(",") if f))
for _l in _report:
    print(_l)
shim = '''

# ---------------------------------------------------------------------------
# `model_file` shim: the runtime imports Model + its args class from THIS file
# (it lives inside the checkpoint). We reuse the registry architecture and swap
# each VQ'd expert module for VQSwitchLinear before weights load.
#
# The base class may live in EITHER runtime, and for most families it lives in
# BOTH under the same name -- see _resolve_arch below and vqlab/arch_resolve.py
# for why the order is decided by the artifact's modalities. Hardcoding mlx_lm
# here shipped a glm5_next artifact that passed check-release AND check-bundle
# and then died on import with ModuleNotFoundError (2026-08-30); resolving it
# mlx_lm-FIRST then bound 17 multimodal bundles to a text-only arch until
# 2026-09-19. Neither gate executes the bundle, which is what `smoke` is for.
# ---------------------------------------------------------------------------
import importlib as _importlib
import json as _json
import pathlib as _pathlib

_cfg = _json.load(open(_pathlib.Path(__file__).parent / "config.json"))
''' + _ARCH_PRELUDE + _ARCH_COERCE + _ARCH_PATHWALK + _ARCH_VLM_SANITIZE + _ARCH_ARRAYISH + '''


class Model(_arch.Model):
    def __init__(self, args):
        args = _coerce_module_configs(args)
        super().__init__(args)
        for _path, _m in _cfg.get("vq_modules", {}).items():
            _obj, _leaf = _reach_vq(self, _path)
            _parts = [_leaf]
            _pb = _m.get("pack_bits", 0)
            if _pb:
                # packed: uint32 words, 32 codes per BITS words, row-local
                _nsub = _m["in"] // _m["dim"]
                _ncol = (_nsub + 31) // 32 * _pb
                _ct = mx.uint32
            else:
                _ncol = _m["in"] // _m["dim"]
                _ct = mx.uint8 if _m["k"] <= 256 else mx.uint16
            _attach_vq(_obj, _parts[-1], VQSwitchLinear(
                mx.zeros((_m["experts"], _m["out"], _ncol), dtype=_ct),
                mx.zeros((_m["k"], _m["dim"]), dtype=mx.float16),
                mx.zeros((_m["experts"], _m["out"], _m["in"] // _m["group"]),
                         dtype=mx.float16),
                group_size=_m["group"],
                pack_bits=_pb,
                in_features=_m["in"] if _pb else None,
            ))
        _ple = _cfg.get("vq_ple")
        if _ple:
            _g = _ple["geometry"]
            for _key in _ple["keys"]:
                _rows, _cols = _ple["shapes"][_key]
                _obj, _leaf = _reach_vq(self, _key)
                _parts = [_leaf]
                _rb = _g.get("row_bytes")
                _codes0 = (mx.zeros((_rows, _rb), dtype=mx.uint8) if _rb else
                           mx.zeros((_rows, _cols // _g["dim"]), dtype=mx.uint16))
                _attach_vq(_obj, _parts[-1], VQPLEEmbedding(
                    _codes0,
                    mx.zeros((_g["k"], _g["dim"]), dtype=mx.float16),
                    mx.zeros((_rows, _cols // _g["group"]), dtype=mx.float16),
                    group_size=_g["group"],
                    packed_nsub=(_cols // _g["dim"]) if _rb else 0,
                ))

    def __call__(self, *_a, **_kw):
        return _arrayish(super().__call__(*_a, **_kw))

    def sanitize(self, _weights):
        if _LOADER == "mlx_vlm" and _arch.__name__.startswith("mlx_vlm."):
            return _sanitize_for_vlm(self, _weights)
        _base = getattr(super(), "sanitize", None)
        return _base(_weights) if _base is not None else _weights
'''
_model_py = runtime + shim
# NEVER ship a model.py that cannot parse. The dense bundler has always done
# this; this one did not, and wrote a bundle with a SyntaxError in it that was
# only caught when a smoke run came back silent.
compile(_model_py, "model.py", "exec")
(ART / "model.py").write_text(_model_py)
print(f"wrote model.py + config keys: {len(vq_modules)} vq modules -> {ART}")
