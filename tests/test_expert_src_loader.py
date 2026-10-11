"""core.source.expert_src's loader on synthetic checkpoints (CPU only, no GPU).

Moved here from expert_src._selftest (untangle step 9a): it is the only
place expert_src read the FAMILY table, and core may not import family.
Synthesizes a tiny unfused checkpoint (glm5_next layout, experts across two
shards), a fused one (qwen3_5) and a dense one (gemma4_e4b); checks the
loader against hand-built stacks, and gates both directions (III.5): the
unfused reader must FAIL on a gap in the expert numbering, not mis-stack.
"""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from vqlab.core.source.expert_src import count_experts, load_expert_stack  # noqa: E402
from vqlab.family.registry import FAMILY  # noqa: E402

E, OUT, IN = 5, 8, 16


def test_unfused_two_shards(tmp_path):
    with mx.stream(mx.cpu):
        mx.random.seed(3)
        fam_u = FAMILY["glm5_next"]
        w0, w1, wm, ref = {}, {}, {}, {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            key_t, _ = fam_u["proj"][proj]
            ts = [mx.random.normal((OUT, IN)).astype(mx.bfloat16) for _ in range(E)]
            ref[proj] = mx.stack(ts)
            for e, t in enumerate(ts):
                k = fam_u["src_key"].format(li=0, key=key_t, e=e)
                (w0 if e < 3 else w1)[k] = t
                wm[k] = ("model-00001-of-00002.safetensors" if e < 3
                         else "model-00002-of-00002.safetensors")
        u = tmp_path / "unfused"
        u.mkdir()
        mx.save_safetensors(str(u / "model-00001-of-00002.safetensors"), w0)
        mx.save_safetensors(str(u / "model-00002-of-00002.safetensors"), w1)
        for proj in ("gate_proj", "up_proj", "down_proj"):
            T = load_expert_stack(u, wm, fam_u, 0, proj)
            assert T.shape == (E, OUT, IN), proj
            assert bool(mx.all(T == ref[proj]).item()), proj
        T = load_expert_stack(u, wm, fam_u, 0, "gate_proj", experts=2)
        assert T.shape == (2, OUT, IN)
        assert bool(mx.all(T == ref["gate_proj"][:2]).item())
        assert count_experts(fam_u, wm, 0) == E

        bad = {k: v for k, v in wm.items() if ".experts.2." not in k}
        with pytest.raises(ValueError):
            load_expert_stack(u, bad, fam_u, 0, "gate_proj")


def test_fused_halves(tmp_path):
    with mx.stream(mx.cpu):
        mx.random.seed(3)
        fam_f = FAMILY["qwen3_5"]
        gu = mx.random.normal((E, 2 * OUT, IN)).astype(mx.bfloat16)
        dn = mx.random.normal((E, IN, OUT)).astype(mx.bfloat16)
        kg = fam_f["src_key"].format(li=0, key="gate_up_proj")
        kd = fam_f["src_key"].format(li=0, key="down_proj")
        mx.save_safetensors(str(tmp_path / "model-00001-of-00001.safetensors"),
                            {kg: gu, kd: dn})
        wmf = {kg: "model-00001-of-00001.safetensors",
               kd: "model-00001-of-00001.safetensors"}
        assert bool(mx.all(load_expert_stack(tmp_path, wmf, fam_f, 0, "gate_proj")
                           == gu[:, :OUT, :]).item())
        assert bool(mx.all(load_expert_stack(tmp_path, wmf, fam_f, 0, "up_proj")
                           == gu[:, OUT:, :]).item())
        assert bool(mx.all(load_expert_stack(tmp_path, wmf, fam_f, 0, "down_proj")
                           == dn).item())


def test_dense_2d_gets_expert_axis(tmp_path):
    with mx.stream(mx.cpu):
        mx.random.seed(3)
        fam_d = FAMILY["gemma4_e4b"]
        kd2 = fam_d["src_key"].format(li=0, key="gate_proj")
        t2 = mx.random.normal((OUT, IN)).astype(mx.bfloat16)
        mx.save_safetensors(str(tmp_path / "model-00001-of-00001.safetensors"), {kd2: t2})
        T = load_expert_stack(tmp_path, {kd2: "model-00001-of-00001.safetensors"},
                              fam_d, 0, "gate_proj")
        assert T.shape == (1, OUT, IN)
