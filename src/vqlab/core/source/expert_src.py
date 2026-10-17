#!/usr/bin/env python3
"""Shared expert-stack loader — every reader of MoE source weights goes here.

Motivation (GLM-5.3-Flash onboarding, 2026-08-28): glm5_next stores experts
UNFUSED as per-expert 2D tensors (`...mlp.experts.{e}.gate_proj.weight`,
no gate_up stack, no [E, out, in] stack anywhere in the checkpoint — measured
from the safetensors headers). Every reader before this file assumed a single
pre-stacked 3D key; three of them (fit_moe, verify_artifact,
init_sweep) each had their own copy of the read-slice-eval dance. This
module is the one implementation of all source layouts:

  fused 3D      src_key names one [E, 2I, H] tensor, proj map half-slices it
                (qwen3_5 / qwen4_exp)
  stacked 3D    src_key names one [E, out, in] tensor, proj map is direct
                (qwen3_5_mlx, mlx-format conversions)
  dense 2D      one [out, in] tensor, returned as [1, out, in]
                (gemma4_e4b, qwen3_8_dense)
  unfused       src_key contains "{e}": one 2D tensor PER EXPERT, gathered
                and stacked along a new E axis (glm5_next)

All reads are created AND eval'd under mx.stream(mx.cpu) — the IV.1 rule:
a lazy read paid inside a GPU command buffer is a watchdog kill, and the
stream binds at op-creation time, so the load and the slice must both be
built inside the block.

CPU-only loader checks (no GPU, safe on a busy box):

    pytest tests/test_expert_src_loader.py
"""
from __future__ import annotations

import pathlib
import re

import mlx.core as mx

from vqlab.core.artifacts.artifact import read_header


def is_unfused(fam: dict) -> bool:
    return "{e}" in fam["src_key"]


def probe_key(fam: dict, li: int) -> str:
    """A representative source key for existence checks (expert 0)."""
    key_t, _ = fam["proj"]["down_proj"]
    return fam["src_key"].format(li=li, key=key_t, e=0)


def count_experts(fam: dict, src_idx: dict, li: int) -> int:
    """Number of experts on one layer, discovered from the index — never
    trusted from a config (the config and the checkpoint can disagree; the
    index describes the bytes we will actually read)."""
    key_t, _ = fam["proj"]["down_proj"]
    sentinel = "EXPERTIDX"                      # survives re.escape verbatim
    pat = re.escape(fam["src_key"].format(li=li, key=key_t, e=sentinel)
                    ).replace(sentinel, r"(\d+)")
    es = {int(m.group(1)) for k in src_idx if (m := re.fullmatch(pat, k))}
    if not es:
        raise KeyError(f"no expert keys match layer {li} for this family")
    if es != set(range(len(es))):
        raise ValueError(f"layer {li}: expert indices not contiguous 0..E-1 "
                         f"(got {len(es)} indices, max {max(es)})")
    return len(es)


def _raw_tensor(path: str, key: str, hdr_cache: dict) -> mx.array:
    """One tensor's bytes as uint8 [*shape] (1-byte dtypes only), read from
    the safetensors header + offsets. mx.load refuses a WHOLE shard that
    holds any F8_E8M0 tensor (DeepSeek-V4's FP4 scales), so the mxfp4 path
    reads its pairs this way instead."""
    import numpy as np
    if path not in hdr_cache:
        hdr_cache[path] = read_header(path, with_base=True)
    h, base = hdr_cache[path]
    m = h[key]
    if m["dtype"] not in ("I8", "U8", "F8_E8M0", "F8_E4M3"):
        raise ValueError(f"{key}: raw read is for 1-byte dtypes, got {m['dtype']}")
    a, b = m["data_offsets"]
    arr = np.memmap(path, np.uint8, "r", offset=base + a, shape=(b - a,))
    return mx.array(np.asarray(arr).reshape(m["shape"]))


def _dequant_mxfp4(w8: mx.array, s8: mx.array, key: str) -> mx.array:
    """One FP4 expert matrix -> bf16, exactly. `w8` holds two e2m1 codes per
    byte ([out, in/2]); `s8` one E8M0 exponent per 32 inputs ([out, in/32]).
    mx.dequantize's mxfp4 mode reads the same nibble order and exponent
    bias, so this is a reinterpretation, not a re-quantization."""
    if w8.shape[-1] % 4 or s8.shape[-1] * 16 != w8.shape[-1]:
        raise ValueError(f"{key}: not an mxfp4 pair (weight {w8.shape}, "
                         f"scale {s8.shape})")
    return mx.dequantize(mx.view(w8, dtype=mx.uint32), s8, group_size=32,
                         bits=4, mode="mxfp4").astype(mx.bfloat16)


def load_expert_stack(src_dir, src_idx: dict, fam: dict, li: int, proj: str,
                      experts: int | None = None, shard_path=None,
                      shard_cache: dict | None = None) -> mx.array:
    """Return one projection's weights for one layer as [E, out, in] fp/bf16.

    experts     read only the first N experts (unfused layout reads ONLY
                those tensors — the cheap path init_sweep needs);
                for fused/stacked layouts the stack is sliced after load.
    shard_path  optional fn(shard_filename) -> local path (the fitter's
                staging hook). Default reads src_dir/<shard>.
    shard_cache optional dict for one-shard-resident caching across calls
                (verify_artifact's pattern: cleared so one shard lives).
    """
    src_dir = pathlib.Path(src_dir)
    key_t, half = fam["proj"][proj]

    def _resolve(sh):
        return str(shard_path(sh)) if shard_path else str(src_dir / sh)

    def _shard(sh):
        if shard_cache is None:
            return mx.load(_resolve(sh))
        if sh not in shard_cache:
            shard_cache.clear()                 # one src shard resident
            shard_cache[sh] = mx.load(_resolve(sh))
        return shard_cache[sh]

    with mx.stream(mx.cpu):
        if is_unfused(fam):
            if half is not None:
                raise ValueError("unfused layout with a fused-half proj map "
                                 "— family/registry.py entry is inconsistent")
            E = count_experts(fam, src_idx, li)
            if experts is not None:
                E = min(E, experts)
            keys = [fam["src_key"].format(li=li, key=key_t, e=e)
                    for e in range(E)]
            # group by shard so each shard is opened once, but PRESERVE
            # expert order in the stack — order is the routing contract.
            per_expert: list = [None] * E
            by_shard: dict[str, list[int]] = {}
            for e, k in enumerate(keys):
                by_shard.setdefault(src_idx[k], []).append(e)
            if fam.get("src_quant") == "mxfp4":
                hdr: dict = {}
                for e, k in enumerate(keys):
                    sk = k[: -len(".weight")] + ".scale"
                    per_expert[e] = _dequant_mxfp4(
                        _raw_tensor(_resolve(src_idx[k]), k, hdr),
                        _raw_tensor(_resolve(src_idx[sk]), sk, hdr), k)
            else:
                for sh, es in by_shard.items():
                    data = _shard(sh)
                    for e in es:
                        per_expert[e] = data[keys[e]]
            T = mx.stack(per_expert)
        else:
            sk = fam["src_key"].format(li=li, key=key_t)
            T = _shard(src_idx[sk])[sk]
            if T.ndim == 2:                     # dense family: E=1
                T = T[None]
            if half is not None:                # fused gate_up, halves on OUT
                mid = T.shape[1] // 2
                T = T[:, mid * half:mid * (half + 1), :]
            if experts is not None:
                T = T[:experts]
        mx.eval(T)
    return T



if __name__ == "__main__":
    raise SystemExit(__doc__)
