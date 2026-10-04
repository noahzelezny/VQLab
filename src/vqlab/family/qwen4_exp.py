"""qwen4_exp (Qwen3.8-Flash-Next): the reference behaviours VQLab checks.

Flash-Next hashes each token's n-grams into per-layer embedding tables
(PLE). The hash multipliers come from `splitmix64(seed + ...)`; the
checkpoint STORES the ones it was trained with
(`model.layers.<i>.ple.ple_embedding.layer_multipliers`), but the
architecture file discards them and rebuilds them from the config's `seed`
-- whose default was 0 in the mlx-lm fork Knurlogic vendored, against the
reference's 1234 (2026-10-03). Every Flash-Next model, ours and Qwen's own,
read the wrong rows on every token, and nothing compared the rebuilt
multipliers with the stored ones. `ple_multiplier_check` does.
"""
from __future__ import annotations

import json
import pathlib
import re
import struct

from vqlab.family import FamilyPlugin, Parity

_KEY = re.compile(r"^model\.layers\.(\d+)\.ple\.ple_embedding\.layer_multipliers$")


def _stored(model_dir: pathlib.Path) -> dict:
    """{layer: [int, ...]} read raw from the shards (headers + a few bytes)."""
    import numpy as np
    idx = json.load(open(model_dir / "model.safetensors.index.json"))["weight_map"]
    out = {}
    for k, f in idx.items():
        m = _KEY.match(k)
        if not m:
            continue
        with open(model_dir / f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            e = json.loads(fh.read(n))[k]
            a, b = e["data_offsets"]
            fh.seek(8 + n + a)
            out[int(m.group(1))] = [int(x) for x in np.frombuffer(fh.read(b - a), dtype=np.int64)]
    return out


def _default(cls, name):
    """A dataclass field's default, whether declared plain or as a factory."""
    import dataclasses
    f = cls.__dataclass_fields__[name]
    return f.default_factory() if f.default is dataclasses.MISSING else f.default


def rebuilt(model_dir: pathlib.Path) -> dict:
    """{layer: [int, ...]} as the LOADED architecture rebuilds them from this
    config (its own TextArgs default fills a missing `seed`)."""
    import importlib
    import vqlab  # noqa: F401  serves knurlogic's architecture as mlx_lm.models.qwen4_exp
    A = importlib.import_module("mlx_lm.models.qwen4_exp")
    cfg = json.load(open(model_dir / "config.json"))
    t = cfg.get("text_config", cfg)
    seed = t.get("seed", _default(A.TextArgs, "seed"))
    ids = t.get("ple_layer_ids", _default(A.TextArgs, "ple_layer_ids"))
    v, n = t["vocab_size"], t["ngram_size"]
    half = max(1, (((1 << 63) - 1) // max(v, 1)) // 2)
    out = {}
    for li in range(t["num_hidden_layers"]):
        if li + 1 not in ids:
            continue
        base = seed + A._PRIME_1 * ids.index(li + 1)
        out[li] = [2 * (A._splitmix64((base + A._GAMMA * (i + 1)) & A._MASK64) % half) + 1
                   for i in range(n)]
    return out


def used(model_dir: pathlib.Path) -> dict:
    """{layer: [int, ...]} the multipliers the LOADED runtime actually hashes
    with: the model is built lazily from this config (nothing allocated), the
    checkpoint's stored multiplier tensors go through the architecture's own
    sanitize (knurlogic edit 4 adopts them there), and each PLE module's
    `_mults` is read back. This tests the path that runs, not a formula."""
    import importlib
    import mlx.core as mx
    import vqlab  # noqa: F401
    A = importlib.import_module("mlx_lm.models.qwen4_exp")
    cfg = json.load(open(model_dir / "config.json"))
    model = A.Model(A.ModelArgs.from_dict(cfg))
    stored = {f"model.layers.{li}.ple.ple_embedding.layer_multipliers": mx.array(v, dtype=mx.int64)
              for li, v in _stored(model_dir).items()}
    target = model.language_model if hasattr(model, "language_model") else model
    san = getattr(target, "sanitize", None) or model.sanitize
    san(dict(stored))
    out = {}
    for li, layer in enumerate(target.model.layers if hasattr(target, "model") else target.layers):
        ple = getattr(layer, "ple", None)
        if ple is not None:
            out[li] = [int(x) for x in ple.ple_embedding._mults.tolist()]
    return out


def ple_multiplier_check(model_dir, runtime: bool = True) -> list:
    """[(layer, stored, runtime)] for every PLE layer whose multipliers in the
    LOADED runtime (runtime=True; the formula rebuild with False) differ from
    the checkpoint's stored ones; [] when they agree (or none are stored)."""
    d = pathlib.Path(model_dir)
    s = _stored(d)
    r = used(d) if runtime else rebuilt(d)
    return [(li, s[li], r.get(li)) for li in sorted(s) if s[li] != r.get(li)]


SPEC = FamilyPlugin(
    name="qwen4_exp",
    model_types=("qwen4_exp",),
    notes="Qwen3.8-Flash-Next; architecture from knurlogic (stock mlx-lm 0.32 has none)",
    reference="Qwen/Qwen3.8-Flash-Next modeling code; the checkpoint's own stored "
              "ple_embedding.layer_multipliers",
    parity=(
        Parity(
            name="ple_hash_multipliers",
            reference="n-gram hash multipliers = splitmix64(seed=1234 + PRIME*layer + GAMMA*i); "
                      "the trained values are STORED in the checkpoint (layer_multipliers)",
            runtime="the architecture rebuilds them from config `seed` (default was 0 in the "
                    "fork; knurlogic family-audit fixes it to 1234); "
                    "vqlab.family.qwen4_exp.ple_multiplier_check compares rebuilt vs stored",
            test="tests/test_qwen4_exp_parity.py::test_ple_multipliers_match_stored",
        ),
    ),
)
