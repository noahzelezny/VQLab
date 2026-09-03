"""VQEmbedding must hand back the MODEL's dtype, not its decode dtype.

Why this file exists. VQEmbedding decodes in fp16 (fp16 codebook gather x
fp16 scales) for speed and footprint. It used to RETURN that fp16, and
nothing caught it because fp16 is a perfectly good number -- the damage is
downstream and dtype-level, not value-level:

    mlx promotes bfloat16 (+,*) float16 -> FLOAT32

so one fp16 per-layer-embedding output turned every activation after it --
h, q/k/v, and the attention mask -- fp32 for the rest of the forward. On
gemma-4-e4b-it-VQ-PLE that reached mx.fast.scaled_dot_product_attention at
head_dim 256/512 in fp32 with an fp32 mask, whose steel_attention kernel
wants 53760 B of threadgroup memory against Metal's 32768 B cap:

    RuntimeError: [metal::Device] Unable to load kernel
    steel_attention_float32_bq32_bk16_bd256_..._maskfloat32_...
    Threadgroup memory size (53760) exceeds the maximum (32768)

Short prompts survived (they take the vector kernel, which fits); anything
chat-templated did not. VQPLEEmbedding in vq_switch has always cast on the
way out -- the dense PLE path is the one that lost the rule, so the test
that guards it lives here and is about DTYPE, not tolerance.
"""
import pathlib
import sys

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src" / "vqlab"))
import vq_dense as D


def _embed(rows=32, IN=64, K=16, dim=2, group=16, seed=0, **kw):
    rng = np.random.default_rng(seed)
    nsub = IN // dim
    codes = mx.array(rng.integers(0, K, size=(rows, nsub)).astype(np.uint8))
    codebook = mx.array(rng.standard_normal((K, dim)).astype(np.float16))
    scales = mx.array(
        (0.1 * rng.standard_normal((rows, IN // group))).astype(np.float16))
    return D.VQEmbedding(codes, codebook, scales, group_size=group, **kw)


def test_defaults_to_bfloat16():
    """The default is the dtype mlx-lm actually computes gemma in."""
    out = _embed()(mx.array([[0, 1, 2]]))
    assert out.dtype == mx.bfloat16
    assert out.shape == (1, 3, 64)


@pytest.mark.parametrize("dt", [mx.bfloat16, mx.float16, mx.float32])
def test_out_dtype_is_honoured(dt):
    """An fp16 or fp32 artifact must be able to stay in its own dtype."""
    assert _embed(out_dtype=dt)(mx.array([[0, 1]])).dtype == dt


def test_no_float32_promotion_against_a_bf16_stack():
    """THE regression. Adding the embedding output to a bf16 activation must
    stay bf16 -- fp16 here is what silently produced an fp32 forward."""
    out = _embed()(mx.array([[0, 1, 2]]))
    h = mx.zeros(out.shape, dtype=mx.bfloat16)
    assert (h + out).dtype == mx.bfloat16
    assert (h * out).dtype == mx.bfloat16


def test_cast_is_the_only_change_to_the_values():
    """The fix must be a cast, not a different decode: the returned array is
    exactly the fp16 decode rounded once to the model dtype."""
    ids = mx.array([[0, 5, 7]])
    fp16 = _embed(out_dtype=mx.float16)(ids)
    bf16 = _embed()(ids)
    assert mx.array_equal(bf16, fp16.astype(mx.bfloat16))
