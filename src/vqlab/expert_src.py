#!/usr/bin/env python3
"""Shared expert-stack loader — every reader of MoE source weights goes here.

Motivation (GLM-5.3-Flash onboarding, 2026-08-28): glm5_next stores experts
UNFUSED as per-expert 2D tensors (`...mlp.experts.{e}.gate_proj.weight`,
no gate_up stack, no [E, out, in] stack anywhere in the checkpoint — measured
from the safetensors headers). Every reader before this file assumed a single
pre-stacked 3D key; three of them (vq_397b_codes, verify_artifact,
probe_init_sweep) each had their own copy of the read-slice-eval dance. This
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

CPU-only selftest (no GPU, safe on a busy box):

    python -m vqlab.expert_src --selftest
"""
from __future__ import annotations

import pathlib
import re
import sys

import mlx.core as mx

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from families import FAMILY


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


def load_expert_stack(src_dir, src_idx: dict, fam: dict, li: int, proj: str,
                      experts: int | None = None, shard_path=None,
                      shard_cache: dict | None = None) -> mx.array:
    """Return one projection's weights for one layer as [E, out, in] fp/bf16.

    experts     read only the first N experts (unfused layout reads ONLY
                those tensors — the cheap path probe_init_sweep needs);
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
                                 "— families.py entry is inconsistent")
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


# --------------------------------------------------------------------------
# CPU-only selftest: synthesizes a tiny fused checkpoint and a tiny unfused
# one, checks the loader agrees with a hand-built stack on both, and gates
# both directions (III.5): the unfused reader must FAIL on a gap in the
# expert numbering, not silently mis-stack.
# --------------------------------------------------------------------------

def _selftest() -> int:
    import tempfile
    E, OUT, IN = 5, 8, 16
    ok = True

    def report(name, good, detail=""):
        nonlocal ok
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {name}"
              f"{(' — ' + detail) if detail else ''}")

    with mx.stream(mx.cpu):
        mx.random.seed(3)
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="vqlab_expert_src_"))

        # --- unfused (glm5_next layout), experts split across two shards
        fam_u = FAMILY["glm5_next"]
        w0, w1, wm = {}, {}, {}
        ref = {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            key_t, _ = fam_u["proj"][proj]
            ts = [mx.random.normal((OUT, IN)).astype(mx.bfloat16)
                  for _ in range(E)]
            ref[proj] = mx.stack(ts)
            for e, t in enumerate(ts):
                k = fam_u["src_key"].format(li=0, key=key_t, e=e)
                (w0 if e < 3 else w1)[k] = t
                wm[k] = ("model-00001-of-00002.safetensors" if e < 3
                         else "model-00002-of-00002.safetensors")
        u = tmp / "unfused"; u.mkdir()
        mx.save_safetensors(str(u / "model-00001-of-00002.safetensors"), w0)
        mx.save_safetensors(str(u / "model-00002-of-00002.safetensors"), w1)
        for proj in ("gate_proj", "up_proj", "down_proj"):
            T = load_expert_stack(u, wm, fam_u, 0, proj)
            good = (T.shape == (E, OUT, IN)
                    and bool(mx.all(T == ref[proj]).item()))
            report(f"unfused stack {proj}", good, f"shape {T.shape}")
        T = load_expert_stack(u, wm, fam_u, 0, "gate_proj", experts=2)
        report("unfused experts=2 reads only 2",
               T.shape == (2, OUT, IN)
               and bool(mx.all(T == ref["gate_proj"][:2]).item()))
        report("count_experts", count_experts(fam_u, wm, 0) == E)

        # known-bad: a gap in expert numbering must raise, not mis-stack
        bad = {k: v for k, v in wm.items() if ".experts.2." not in k}
        try:
            load_expert_stack(u, bad, fam_u, 0, "gate_proj")
            report("gap in expert indices FAILS", False, "loader accepted it")
        except ValueError:
            report("gap in expert indices FAILS", True)

        # --- fused (qwen3_5 layout): loader must match the half-slice
        fam_f = FAMILY["qwen3_5"]
        gu = mx.random.normal((E, 2 * OUT, IN)).astype(mx.bfloat16)
        dn = mx.random.normal((E, IN, OUT)).astype(mx.bfloat16)
        f = tmp / "fused"; f.mkdir()
        kg = fam_f["src_key"].format(li=0, key="gate_up_proj")
        kd = fam_f["src_key"].format(li=0, key="down_proj")
        mx.save_safetensors(str(f / "model-00001-of-00001.safetensors"),
                            {kg: gu, kd: dn})
        wmf = {kg: "model-00001-of-00001.safetensors",
               kd: "model-00001-of-00001.safetensors"}
        report("fused gate half", bool(mx.all(
            load_expert_stack(f, wmf, fam_f, 0, "gate_proj") == gu[:, :OUT, :]
        ).item()))
        report("fused up half", bool(mx.all(
            load_expert_stack(f, wmf, fam_f, 0, "up_proj") == gu[:, OUT:, :]
        ).item()))
        report("fused down direct", bool(mx.all(
            load_expert_stack(f, wmf, fam_f, 0, "down_proj") == dn).item()))

        # --- dense 2D gets the E=1 axis
        fam_d = FAMILY["gemma4_e4b"]
        kd2 = fam_d["src_key"].format(li=0, key="gate_proj")
        d = tmp / "dense"; d.mkdir()
        t2 = mx.random.normal((OUT, IN)).astype(mx.bfloat16)
        mx.save_safetensors(str(d / "model-00001-of-00001.safetensors"),
                            {kd2: t2})
        T = load_expert_stack(d, {kd2: "model-00001-of-00001.safetensors"},
                              fam_d, 0, "gate_proj")
        report("dense 2D -> [1,out,in]", T.shape == (1, OUT, IN))

    print(("all checks passed" if ok else "CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    raise SystemExit(__doc__)
