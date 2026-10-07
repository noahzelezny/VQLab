"""stage-bandwidth: bytes billed per decode stage, joined to a timeline.

Synthetic header-only safetensors: no tensor data is ever read, so a
header that declares sizes is a complete artifact for this tool."""
import json
import struct

import pytest

from vqlab.bench.serve_timeline import summarize
from vqlab.bench.stage_bandwidth import by_kind, bytes_by_stage, join, stage_of


def _art(tmp_path, tensors, cfg):
    hdr = {}
    off = 0
    for name, (dtype, shape) in tensors.items():
        n = 1
        for d in shape:
            n *= d
        size = n * {"F16": 2, "U8": 1, "F32": 4}[dtype]
        hdr[name] = {"dtype": dtype, "shape": shape,
                     "data_offsets": [off, off + size]}
        off += size
    raw = json.dumps(hdr).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw)
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    return tmp_path


def test_stage_names_match_decode_timeline():
    assert stage_of("model.layers.3.self_attn.q_proj.weight") == "L03.attn"
    assert stage_of("model.layers.3.linear_attn.in_proj.weight") == "L03.attn"
    assert stage_of("model.layers.3.input_layernorm.weight") == "L03.attn"
    assert stage_of("model.layers.3.post_attention_layernorm.weight") == "L03.mlp"
    assert stage_of("model.layers.12.mlp.switch_mlp.up_proj.codes") == "L12.mlp"
    assert stage_of("model.layers.12.mlp.shared_expert.up_proj.weight") == "L12.mlp"
    assert stage_of("language_model.model.layers.0.mlp.gate.weight") == "L00.mlp"
    assert stage_of("model.embed_tokens.weight") == "embed"
    assert stage_of("lm_head.weight") == "lm_head"
    assert stage_of("model.norm.weight") == "final_norm"
    assert stage_of("model.visual.blocks.0.attn.qkv.weight") == "unused"


@pytest.fixture
def moe(tmp_path):
    return _art(tmp_path, {
        "model.embed_tokens.weight": ("F16", [1000, 8]),          # 16000 B
        "model.layers.0.self_attn.q_proj.weight": ("F16", [8, 8]),  # 128
        "model.layers.0.mlp.switch_mlp.up_proj.weight": ("U8", [4, 100]),  # 400
        "model.layers.0.mlp.shared_expert.up_proj.weight": ("F16", [10, 10]),  # 200
        "model.norm.weight": ("F16", [8]),                           # 16
        "lm_head.weight": ("F16", [1000, 8]),                        # 16000
    }, {"num_experts": 4, "num_experts_per_tok": 1})


def test_bytes_are_billed_as_a_decode_step_reads_them(moe):
    b = bytes_by_stage(moe)
    assert b["embed"] == pytest.approx(16000 / 1000)        # one row gathered
    assert b["L00.attn"] == 128
    assert b["L00.mlp"] == pytest.approx(400 / 4 + 200)     # top-1 of 4 + shared
    assert b["final_norm"] == 16
    assert b["lm_head"] == 16000


def test_join_ranks_by_time_above_the_roofline(moe):
    tl = {"stages": [
        {"name": "embed", "kind": None, "ms": 0.01},
        {"name": "L00.gdn", "kind": "gdn", "ms": 1.0},     # gdn maps to the attn half
        {"name": "L00.mlp", "kind": "mlp", "ms": 0.5},
        {"name": "final_norm", "kind": None, "ms": 0.01},
        {"name": "lm_head", "kind": None, "ms": 0.2},
    ]}
    j = join(tl, bytes_by_stage(moe), peak_gbs=1.0)       # 1 GB/s: 1e6 B/ms
    rows = {r["name"]: r for r in j["rows"]}
    assert rows["L00.gdn"]["bytes"] == 128
    assert rows["lm_head"]["floor_ms"] == pytest.approx(0.016)
    assert rows["lm_head"]["excess_ms"] == pytest.approx(0.2 - 0.016)
    assert rows["lm_head"]["gbs"] == pytest.approx(16000 / 0.2e6)
    k = by_kind(j["rows"])
    assert set(k) == {"embed", "gdn", "mlp", "final_norm", "lm_head"}
    assert j["unclaimed"] == {}


def test_bytes_of_an_untimed_layer_are_reported_not_dropped(moe):
    tl = {"stages": [{"name": "lm_head", "kind": None, "ms": 0.2}]}
    j = join(tl, bytes_by_stage(moe), peak_gbs=None)
    assert "L00.mlp" in j["unclaimed"] and "floor_ms" not in j["rows"][0]


def test_serve_timeline_summary_is_a_median_partition():
    runs = [{"spans": {"queue": q, "prefill_forward": p, "decode_forward": 1.0},
             "whole_s": q + p + 1.0, "unaccounted_s": 0.0}
            for q, p in ((0.1, 2.0), (0.3, 2.2), (0.2, 2.1))]
    s = summarize(runs)
    assert s["median_s"]["prefill_forward"] == 2.1
    assert s["min_s"]["queue"] == 0.1 and s["max_s"]["queue"] == 0.3
    assert s["ttft_side_s"] == pytest.approx(0.2 + 2.1)
