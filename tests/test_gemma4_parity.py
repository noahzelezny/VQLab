"""gemma4 reference parity (e4b + 26B-A4B, text + vision), CPU only, tiny shapes.

The reference is the maker's code: transformers 5.18
`models/gemma4/modeling_gemma4.py` (cited below as HF:<line>). torch is not
installed, so each test transcribes that forward in numpy (float64) and
compares knurlogic's architecture -- what `import vqlab` serves as
mlx_lm.models.gemma4_text -- and knurlogic's vision tower against it.
PARITY in vqlab/family/gemma4.py names these tests.

Every MLX op runs on the CPU stream (`with mx.stream(mx.cpu)`, never
mx.set_default_device: it leaks into later tests)."""
import importlib
import json
import pathlib
from vqlab import config as _cfg
import huggingface_hub.constants as _hfc
import types

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
pytest.importorskip("knurlogic")
import vqlab  # noqa: E402,F401  serves knurlogic's architecture as mlx_lm.models.gemma4_text

from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

A = importlib.import_module("mlx_lm.models.gemma4_text")
W4 = importlib.import_module("mlx_lm.models.gemma4")
V = pytest.importorskip("knurlogic.engine.families.gemma4.vision.vision")
VC = pytest.importorskip("knurlogic.engine.families.gemma4.vision.config")
VF = pytest.importorskip("knurlogic.engine.families.gemma4.vision")

EXO = _cfg.models()
HUB = pathlib.Path(_hfc.HF_HUB_CACHE)


@pytest.fixture(autouse=True)
def _cpu():
    with mx.stream(mx.cpu):
        yield


def _np(a):
    return np.array(a.astype(mx.float32), dtype=np.float64)


# ---------------------------------------------------------------- reference math

def ref_rms(x, w=None, eps=1e-6):
    """HF:197-216 Gemma4RMSNorm: x * pow(mean(x^2) + eps, -0.5) [* weight] --
    the weight multiplies as-is (no 1 + weight, unlike Gemma 2/3)."""
    y = x * np.power(np.mean(x * x, -1, keepdims=True) + eps, -0.5)
    return y * w if w is not None else y


def ref_gelu_tanh(x):
    """ACT2FN['gelu_pytorch_tanh'] (config hidden_activation)."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def ref_rotate_half(x):
    """HF:773-777."""
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], -1)


def ref_inv_freq(head_dim, rp):
    """modeling_rope_utils: 'default' = 1/theta^(arange(0,d,2)/d);
    'proportional' (modeling_rope_utils.py:193-262) rotates the first
    int(prf*d//2) frequencies (exponent still over d) and pads the rest with 0."""
    theta = rp["rope_theta"]
    if rp.get("rope_type", "default") == "proportional":
        n = int(rp.get("partial_rotary_factor", 1.0) * head_dim // 2)
        f = 1.0 / theta ** (np.arange(0, 2 * n, 2) / head_dim)
        return np.concatenate([f, np.zeros(head_dim // 2 - n)])
    return 1.0 / theta ** (np.arange(0, head_dim, 2) / head_dim)


def ref_rope(x, pos, inv_freq):
    """HF:780-800 + Gemma4TextRotaryEmbedding (HF:1142-1158): emb = cat(f, f)."""
    f = np.outer(pos, inv_freq)
    emb = np.concatenate([f, f], -1)
    cos, sin = np.cos(emb)[:, None, :], np.sin(emb)[:, None, :]   # [L, 1, D] vs x [L, H, D]
    return x * cos + ref_rotate_half(x) * sin


def ref_attend(q, k, v, mask):
    """eager_attention_forward (HF:814-845) at scaling=1.0 (HF:1177), no
    attention softcap (text attention passes none). q [L,H,D], k/v [S,Hk,D]."""
    rep = q.shape[1] // k.shape[1]
    k, v = np.repeat(k, rep, 1), np.repeat(v, rep, 1)
    s = np.einsum("lhd,shd->hls", q, k)
    s = np.where(mask[None], s, -np.inf)
    s = s - s.max(-1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(-1, keepdims=True)
    return np.einsum("hls,shd->lhd", p, v)


def ref_masks(L, window, block_ids=None, use_bidir=None):
    """{layer_type: [L, L] bool} as Gemma4Model.forward builds them
    (HF:2407-2420): with use_bidirectional_attention == 'vision' and image
    tokens, create_masks_for_vision_model (HF:2080-2141): FULL = causal only;
    SLIDING = AND(sliding_window, OR(causal, blockwise)) -- the window is
    masking_utils.sliding_window_overlay (kv > q - W), the blockwise term
    masking_utils.blockwise_overlay (same block id, id >= 0). Otherwise
    (e4b: use_bidirectional_attention None) plain causal / sliding causal."""
    q, k = np.arange(L)[:, None], np.arange(L)[None]
    causal = k <= q
    win = k > q - window
    if use_bidir == "vision" and block_ids is not None:
        b = np.asarray(block_ids)
        block = (b[:, None] == b[None]) & (b[:, None] >= 0)
        return {"full_attention": causal, "sliding_attention": win & (causal | block)}
    return {"full_attention": causal, "sliding_attention": causal & win}


def ref_router(x, p, top_k, eps=1e-6):
    """HF:1318-1352 Gemma4TextRouter: norm (no scale) * scale * H^-0.5 ->
    proj -> softmax over ALL experts (fp32) -> top-k -> renormalize ->
    * per_expert_scale[idx]."""
    h = ref_rms(x, None, eps) * p["scale"] * x.shape[-1] ** -0.5
    s = h @ p["proj.weight"].T
    pr = np.exp(s - s.max(-1, keepdims=True))
    pr /= pr.sum(-1, keepdims=True)
    idx = np.argsort(-pr, -1)[:, :top_k]
    w = np.take_along_axis(pr, idx, -1)
    w = w / w.sum(-1, keepdims=True)
    return idx, w * p["per_expert_scale"][idx]


def ref_text_forward(cfg, P, ids):
    """Gemma4ForCausalLM forward (text only), float64, batch 1, no cache:
    HF:1560-1709 (Gemma4TextModel), 1355-1441 (decoder layer), 1162-1276
    (attention), 1061-1077 (MLP), 1279-1352 (MoE), 1755-1788 (PLE),
    1864-1867 (final softcap). P = knurlogic's parameters by name."""
    H, L, n = cfg.hidden_size, len(ids), cfg.num_hidden_layers
    g = lambda k: P[k]  # noqa: E731
    E = g("model.embed_tokens.weight")
    h = E[ids] * H ** 0.5                                   # HF:1444-1455, 1576
    pli = None
    if cfg.hidden_size_per_layer_input:
        Pd = cfg.hidden_size_per_layer_input
        tok = (g("model.embed_tokens_per_layer.weight")[ids] * Pd ** 0.5).reshape(L, n, Pd)
        proj = (h @ g("model.per_layer_model_projection.weight").T) * H ** -0.5
        proj = ref_rms(proj.reshape(L, n, Pd), g("model.per_layer_projection_norm.weight"))
        pli = (proj + tok) * 2.0 ** -0.5                    # HF:1777-1788
    first_shared = n - cfg.num_kv_shared_layers
    shared = {}
    pos = np.arange(L)
    masks = ref_masks(L, cfg.sliding_window)
    for i in range(n):
        lt = cfg.layer_types[i]
        pre = f"model.layers.{i}."
        a = pre + "self_attn."
        full = lt == "full_attention"
        hd = cfg.global_head_dim if full and cfg.global_head_dim else cfg.head_dim
        k_eq_v = cfg.attention_k_eq_v and full
        nkv = cfg.num_global_key_value_heads if k_eq_v and cfg.num_global_key_value_heads else cfg.num_key_value_heads
        inv = ref_inv_freq(hd, cfg.rope_parameters[lt])
        res = h
        x = ref_rms(h, g(pre + "input_layernorm.weight"))
        q = (x @ g(a + "q_proj.weight").T).reshape(L, -1, hd)
        q = ref_rope(ref_rms(q, g(a + "q_norm.weight")), pos, inv)
        if i >= first_shared:                               # HF:1236-1241
            k, v = shared[lt]
        else:
            k = (x @ g(a + "k_proj.weight").T).reshape(L, nkv, hd)
            v = k if k_eq_v else (x @ g(a + "v_proj.weight").T).reshape(L, nkv, hd)
            k = ref_rope(ref_rms(k, g(a + "k_norm.weight")), pos, inv)
            v = ref_rms(v)                                  # v_norm: no scale (HF:1197)
            shared[lt] = (k, v)                             # last non-shared layer of the type
        o = ref_attend(q, k, v, masks[lt]).reshape(L, -1) @ g(a + "o_proj.weight").T
        h = res + ref_rms(o, g(pre + "post_attention_layernorm.weight"))
        res = h
        m = pre + "mlp."
        x = ref_rms(h, g(pre + "pre_feedforward_layernorm.weight"))
        x = (ref_gelu_tanh(x @ g(m + "gate_proj.weight").T) * (x @ g(m + "up_proj.weight").T)) \
            @ g(m + "down_proj.weight").T
        if cfg.enable_moe_block:                            # HF:1413-1426
            h1 = ref_rms(x, g(pre + "post_feedforward_layernorm_1.weight"))
            rp = {k: g(pre + "router." + k) for k in ("scale", "proj.weight", "per_expert_scale")}
            idx, w = ref_router(res, rp, cfg.top_k_experts)
            x2 = ref_rms(res, g(pre + "pre_feedforward_layernorm_2.weight"))
            sw = pre + "experts.switch_glu."
            Gw, Uw, Dw = g(sw + "gate_proj.weight"), g(sw + "up_proj.weight"), g(sw + "down_proj.weight")
            h2 = np.zeros_like(x2)
            for t in range(L):
                for j, e in enumerate(idx[t]):
                    y = ref_gelu_tanh(Gw[e] @ x2[t]) * (Uw[e] @ x2[t])
                    h2[t] += w[t, j] * (Dw[e] @ y)
            x = h1 + ref_rms(h2, g(pre + "post_feedforward_layernorm_2.weight"))
        h = res + ref_rms(x, g(pre + "post_feedforward_layernorm.weight"))
        if pli is not None:                                 # HF:1431-1438
            gt = ref_gelu_tanh(h @ g(pre + "per_layer_input_gate.weight").T) * pli[:, i]
            h = h + ref_rms(gt @ g(pre + "per_layer_projection.weight").T,
                            g(pre + "post_per_layer_input_norm.weight"))
        h = h * g(pre + "layer_scalar")                     # HF:1440
    h = ref_rms(h, g("model.norm.weight"))
    logits = h @ E.T
    c = cfg.final_logit_softcapping
    return np.tanh(logits / c) * c                          # HF:1864-1867


# ---------------------------------------------------------------- fixtures

def _args(**kw):
    return A.ModelArgs.from_dict(kw)


E4B_LIKE = dict(  # e4b's structure, shrunk: PLE, KV sharing, proportional full rope
    model_type="gemma4_text", hidden_size=32, num_hidden_layers=6, intermediate_size=48,
    num_attention_heads=2, num_key_value_heads=1, head_dim=16, global_head_dim=32,
    vocab_size=64, vocab_size_per_layer_input=64, hidden_size_per_layer_input=8,
    num_kv_shared_layers=2, sliding_window=4, use_double_wide_mlp=True,
    layer_types=["sliding_attention", "full_attention"] * 3,
    rope_parameters={"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1e6,
                                        "rope_type": "proportional"},
                     "sliding_attention": {"rope_theta": 1e4, "rope_type": "default"}},
    final_logit_softcapping=30.0, tie_word_embeddings=True)

A26B_LIKE = dict(  # 26B-A4B's structure, shrunk: MoE + dense MLP, K=V on full layers
    model_type="gemma4_text", hidden_size=32, num_hidden_layers=4, intermediate_size=24,
    num_attention_heads=4, num_key_value_heads=2, num_global_key_value_heads=1,
    head_dim=8, global_head_dim=16, attention_k_eq_v=True, vocab_size=64,
    hidden_size_per_layer_input=0, num_kv_shared_layers=0, sliding_window=4,
    use_double_wide_mlp=False, enable_moe_block=True, num_experts=6, top_k_experts=2,
    moe_intermediate_size=16,
    layer_types=["sliding_attention", "sliding_attention", "full_attention", "full_attention"],
    rope_parameters={"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1e6,
                                        "rope_type": "proportional"},
                     "sliding_attention": {"rope_theta": 1e4, "rope_type": "default"}},
    final_logit_softcapping=30.0, tie_word_embeddings=True)


def _randomize(model, seed=1234):
    """Every parameter random (norm weights and scalars around 1, so a
    1 + weight convention would show)."""
    rng = np.random.default_rng(seed)
    flat = []
    for k, v in tree_flatten(model.parameters()):
        if any(s in k for s in ("norm", "layer_scalar", "scale")):
            a = 1.0 + 0.3 * rng.standard_normal(v.shape)
        else:
            a = rng.standard_normal(v.shape) * (0.6 if "embed" in k else 0.25)
        flat.append((k, mx.array(a.astype(np.float32))))
    model.update(tree_unflatten(flat))
    mx.eval(model.parameters())
    return {k: _np(v) for k, v in tree_flatten(model.parameters())}


def _text_case(kw, seed=1234, L=9):
    args = _args(**kw)
    m = A.Model(args)
    P = _randomize(m, seed)
    ids = np.random.default_rng(seed + 1).integers(1, args.vocab_size, L)
    got = _np(m(mx.array(ids[None]))[0])
    return args, P, ids, got


# ---------------------------------------------------------------- text tests

def test_rmsnorm_plain_weight_and_eps():
    """HF:197-216 vs nn.RMSNorm (every text norm, gemma4_text.py:214-217,
    287-296, 406) and RMSNormNoScale (v_norm, gemma4_text.py:73-81)."""
    rng = np.random.default_rng(1234)
    x = rng.standard_normal((3, 5, 16)).astype(np.float32) * 1e-3   # eps visible at this scale
    w = (1 + 0.5 * rng.standard_normal(16)).astype(np.float32)
    n = nn.RMSNorm(16, eps=1e-6)
    n.weight = mx.array(w)
    np.testing.assert_allclose(_np(n(mx.array(x))), ref_rms(x.astype(np.float64), w), rtol=2e-5)
    np.testing.assert_allclose(_np(A.RMSNormNoScale(16, 1e-6)(mx.array(x))),
                               ref_rms(x.astype(np.float64)), rtol=2e-5)
    at = A.Attention(_args(**E4B_LIKE), 0)
    assert at.q_norm.eps == at.k_norm.eps == at.v_norm.eps == 1e-6
    assert not hasattr(at.v_norm, "weight")


def test_mlp_gelu_tanh():
    """HF:1061-1077 down(gelu_pytorch_tanh(gate x) * up x) vs MLP/geglu
    (gemma4_text.py:94-114, nn.gelu_approx), incl. the double-wide MLP on
    KV-shared layers (HF:1064-1067 vs gemma4_text.py:102-107)."""
    rng = np.random.default_rng(1234)
    g = (rng.standard_normal((4, 64)) * 3).astype(np.float32)
    u = rng.standard_normal((4, 64)).astype(np.float32)
    np.testing.assert_allclose(_np(A.geglu(mx.array(g), mx.array(u))),
                               ref_gelu_tanh(g.astype(np.float64)) * u, rtol=1e-5, atol=1e-6)
    args = _args(**E4B_LIKE)
    sizes = [A.MLP(args, i).gate_proj.weight.shape[0] for i in range(args.num_hidden_layers)]
    first = args.num_hidden_layers - args.num_kv_shared_layers
    assert sizes == [args.intermediate_size * (2 if i >= first else 1) for i in range(len(sizes))]


def test_rope_per_layer_type():
    """Full layers: proportional rope, theta 1e6, partial 0.25 over
    global_head_dim (modeling_rope_utils.py:193-262); sliding: default rope,
    theta 1e4 over head_dim. Runtime: Attention.rope via initialize_rope
    (gemma4_text.py:219-229, mlx_lm rope_utils ProportionalRoPE)."""
    args = _args(**E4B_LIKE)
    rng = np.random.default_rng(1234)
    for li in (0, 1):
        at = A.Attention(args, li)
        lt = args.layer_types[li]
        x = rng.standard_normal((1, 2, 7, at.head_dim)).astype(np.float32)   # [B, H, L, D]
        got = _np(at.rope(mx.array(x), offset=3))[0]
        want = ref_rope(x[0].transpose(1, 0, 2).astype(np.float64), np.arange(3, 10),
                        ref_inv_freq(at.head_dim, args.rope_parameters[lt])).transpose(1, 0, 2)
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4, err_msg=lt)


def test_attention_scale_and_kv_layout():
    """scaling = 1.0 (HF:1177 vs gemma4_text.py:203); full layers use
    global_head_dim and, with attention_k_eq_v, num_global_key_value_heads
    and no v_proj (HF:1173-1209 vs gemma4_text.py:185-212)."""
    args = _args(**A26B_LIKE)
    s, f = A.Attention(args, 0), A.Attention(args, 2)
    assert s.scale == f.scale == 1.0
    assert (s.head_dim, s.n_kv_heads, s.use_k_eq_v) == (8, 2, False)
    assert (f.head_dim, f.n_kv_heads, f.use_k_eq_v) == (16, 1, True)
    assert not hasattr(f, "v_proj") and hasattr(s, "v_proj")


def test_kv_sharing_source_layers():
    """HF:1182-1188 + 1236-1253: a KV-shared layer reads the K/V stored by the
    LAST non-shared layer of its own type. Runtime: previous_kvs
    (gemma4_text.py:433-442)."""
    for kw in (E4B_LIKE, dict(E4B_LIKE, num_hidden_layers=10, num_kv_shared_layers=4,
                              layer_types=["sliding_attention"] * 4 + ["full_attention"]
                              + ["sliding_attention"] * 4 + ["full_attention"])):
        args = _args(**kw)
        tm = A.Gemma4TextModel(args)
        lt = args.layer_types
        first = args.num_hidden_layers - args.num_kv_shared_layers
        prev = lt[:first]
        for j in range(first, args.num_hidden_layers):
            src = len(prev) - 1 - prev[::-1].index(lt[j])
            assert tm.previous_kvs[j] == src, (j, tm.previous_kvs[j], src)


def test_router_topk_softmax():
    """HF:1318-1352 vs Router (gemma4_text.py:117-143): top-k of the raw
    scores then softmax over the k == softmax over all then renormalize."""
    args = _args(**A26B_LIKE)
    r = A.Router(args)
    rng = np.random.default_rng(1234)
    p = {"scale": (1 + 0.3 * rng.standard_normal(32)), "proj.weight": rng.standard_normal((6, 32)),
         "per_expert_scale": 1 + 0.3 * rng.standard_normal(6)}
    r.update(tree_unflatten([(k, mx.array(v.astype(np.float32))) for k, v in p.items()]))
    x = rng.standard_normal((5, 32)).astype(np.float32)
    idx, w = r(mx.array(x))
    idx, w = np.array(idx), _np(w)
    ridx, rw = ref_router(x.astype(np.float64), {k: v.astype(np.float32).astype(np.float64)
                                                 for k, v in p.items()}, 2)
    order = np.argsort(idx, -1)
    np.testing.assert_array_equal(np.take_along_axis(idx, order, -1), np.sort(ridx, -1))
    rorder = np.argsort(ridx, -1)
    np.testing.assert_allclose(np.take_along_axis(w, order, -1),
                               np.take_along_axis(rw, rorder, -1), rtol=1e-4)


def test_e4b_like_forward_matches_reference():
    """End to end on e4b's structure: embedding * sqrt(H), PLE (lookup *
    sqrt(P), projection * H^-0.5, norm, (a + b) * 2^-0.5, per-layer gate),
    KV sharing, double-wide MLP, proportional rope, layer_scalar, final
    softcap 30 -- knurlogic Model logits vs ref_text_forward."""
    args, P, ids, got = _text_case(E4B_LIKE)
    want = ref_text_forward(args, P, ids)
    np.testing.assert_allclose(got, want, rtol=2e-3, atol=2e-3)
    assert np.abs(want).max() > 5      # the softcap is in its nonlinear range


def test_26b_like_forward_matches_reference():
    """End to end on 26B-A4B's structure: dense MLP + MoE in parallel
    (router on the pre-norm residual), K=V full layers with their own KV
    head count, sliding window."""
    args, P, ids, got = _text_case(A26B_LIKE)
    want = ref_text_forward(args, P, ids)
    np.testing.assert_allclose(got, want, rtol=2e-3, atol=2e-3)


# ---------------------------------------------------------------- image mask

L_IMG, WIN = 12, 4
BLOCKS = [-1, -1, -1, 0, 0, 0, 0, 0, 0, -1, 1, 1]   # two images; image 0 longer than the window
LTYPES = ["sliding_attention", "full_attention"]


def _as_bool(m, L):
    if m is None:
        return np.ones((L, L), bool)
    if isinstance(m, str):
        assert m == "causal"
        return np.tril(np.ones((L, L), bool))
    a = np.array(m)
    return a.reshape(a.shape[-2:]).astype(bool)


def _fake_model(bidirectional=None):
    """The attributes Gemma4TextModel._make_masks reads, and the config field
    the corrected runtime gates the image overlay on (None on e2b/e4b,
    "vision" on 26B/31B)."""
    return types.SimpleNamespace(
        layers=[types.SimpleNamespace(layer_type=t) for t in LTYPES], window_size=WIN,
        config=types.SimpleNamespace(use_bidirectional_attention=bidirectional),
        args=types.SimpleNamespace(use_bidirectional_attention=bidirectional),
        use_bidirectional_attention=bidirectional)


def knurlogic_masks(block_ids, bidirectional=None):
    fake = _fake_model(bidirectional)
    h = mx.zeros((1, L_IMG, 4))
    mm = mx.array(block_ids, dtype=mx.int32)[None]
    ms = A.Gemma4TextModel._make_masks(fake, h, [None] * len(LTYPES), mm_mask=mm)
    return {t: _as_bool(m, L_IMG) for t, m in zip(LTYPES, ms)}


def _mask_diff(got, want):
    out = {}
    for t in LTYPES:
        extra = np.argwhere(got[t] & ~want[t])
        missing = np.argwhere(~got[t] & want[t])
        out[t] = (len(extra), len(missing), extra[:4].tolist(), missing[:4].tolist())
    return out


def test_text_only_masks_match_reference():
    """No image: causal on full layers, kv > q - W on sliding layers (both runtimes)."""
    fake = _fake_model(None)
    ms = A.Gemma4TextModel._make_masks(fake, mx.zeros((1, L_IMG, 4)), [None, None], mm_mask=None)
    want = ref_masks(L_IMG, WIN)
    for t, m in zip(LTYPES, ms):
        np.testing.assert_array_equal(_as_bool(m, L_IMG), want[t], err_msg=t)


def test_image_mask_26b_bidirectional_on_sliding_layers():
    """26B-A4B (use_bidirectional_attention='vision'): the reference makes
    image blocks bidirectional on the SLIDING layers and leaves FULL layers
    causal (HF:2080-2141). Knurlogic 6a2ab49 `_make_masks`
    (gemma4_text.py:503-566) does the opposite. Known bug, unfixed."""
    got, want = knurlogic_masks(BLOCKS, "vision"), ref_masks(L_IMG, WIN, BLOCKS, "vision")
    d = _mask_diff(got, want)
    assert all(v[0] == 0 and v[1] == 0 for v in d.values()), (
        "image-block mask differs from the reference "
        "{layer_type: (n_extra_allowed, n_missing, first_extra[q,k], first_missing[q,k])}: "
        f"{d}")


def test_image_mask_e4b_stays_causal():
    """e4b (use_bidirectional_attention=None): the reference uses plain
    causal masks even with images (HF:2408-2420, create_masks_for_generate).
    Knurlogic's ModelArgs has no use_bidirectional_attention, so any mm_mask
    (the vision family always passes one, vision/__init__.py:240-242) widens
    the full layers."""
    got, want = knurlogic_masks(BLOCKS), ref_masks(L_IMG, WIN, BLOCKS, None)
    d = _mask_diff(got, want)
    assert all(v[0] == 0 and v[1] == 0 for v in d.values()), (
        "image-prompt mask differs from the reference "
        "{layer_type: (n_extra_allowed, n_missing, first_extra[q,k], first_missing[q,k])}: "
        f"{d}")


# ---------------------------------------------------------------- vision

def ref_vision_rope(x, pos, theta):
    """HF:707-770 axial rope (freqs over head_dim//2, laid out h,h,w,w) applied
    by apply_multidimensional_rope (HF:848-903). x [L, H, D], pos [L, 2]."""
    D = x.shape[-1]
    sd = D // 2
    inv = 1.0 / theta ** (np.arange(0, sd, 2) / sd)
    parts = []
    for d in range(2):
        f = np.outer(pos[:, d], inv)
        emb = np.concatenate([f, f], -1)
        xp = x[..., d * sd:(d + 1) * sd]
        parts.append(xp * np.cos(emb)[:, None] + ref_rotate_half(xp) * np.sin(emb)[:, None])
    return np.concatenate(parts, -1)


def ref_patchify(img, p):
    """image_processing_gemma4.py:88-98 (C,H,W) -> (pH*pW, p*p*C), and the
    (x, y) position grid (:251-258)."""
    C, H, W = img.shape
    t = img.reshape(C, H // p, p, W // p, p).transpose(1, 3, 2, 4, 0).reshape(-1, p * p * C)
    gx, gy = np.meshgrid(np.arange(W // p), np.arange(H // p), indexing="xy")
    return t, np.stack([gx, gy], -1).reshape(-1, 2)


def ref_pool(h, pos, length):
    """HF:624-689 Gemma4VisionPooler: k*k average by position, * sqrt(hidden)."""
    k = int((h.shape[0] // length) ** 0.5)
    max_x = pos[:, 0].max() + 1
    ki = pos // k
    ki = ki[:, 0] + (max_x // k) * ki[:, 1]
    w = np.eye(length)[ki] / k ** 2
    return (w.T @ h) * h.shape[-1] ** 0.5


def ref_vision_forward(cfg, P, img):
    """Gemma4VisionModel.forward (HF:2009-2050) on one unpadded image:
    patch embed (HF:579-621: 2*(x-0.5), input_proj, + x/y position tables),
    encoder layers (HF:973-1015, attention HF:904-971: q/k norm, v norm
    without scale, 2D rope, scaling 1.0, bidirectional), clipped linears
    (HF:168-194), pooler, standardize."""
    p = cfg.patch_size
    t, pos = ref_patchify(img, p)
    g = lambda k: P[k]  # noqa: E731
    h = (2 * (t - 0.5)) @ g("patch_embedder.input_proj.weight").T
    tab = g("patch_embedder.position_embedding_table")
    h = h + tab[0][pos[:, 0]] + tab[1][pos[:, 1]]

    def lin(pre, x):
        if cfg.use_clipped_linears:
            x = np.clip(x, g(pre + "input_min"), g(pre + "input_max"))
        y = x @ g(pre + "linear.weight").T
        if cfg.use_clipped_linears:
            y = np.clip(y, g(pre + "output_min"), g(pre + "output_max"))
        return y

    L, hd = h.shape[0], cfg.head_dim
    theta = cfg.rope_parameters["rope_theta"]
    for i in range(cfg.num_hidden_layers):
        pre = f"encoder.layers.{i}."
        a = pre + "self_attn."
        x = ref_rms(h, g(pre + "input_layernorm.weight"))
        q = ref_rms(lin(a + "q_proj.", x).reshape(L, -1, hd), g(a + "q_norm.weight"))
        k = ref_rms(lin(a + "k_proj.", x).reshape(L, -1, hd), g(a + "k_norm.weight"))
        v = ref_rms(lin(a + "v_proj.", x).reshape(L, -1, hd))
        q, k = ref_vision_rope(q, pos, theta), ref_vision_rope(k, pos, theta)
        o = ref_attend(q, k, v, np.ones((L, L), bool)).reshape(L, -1)
        h = h + ref_rms(lin(a + "o_proj.", o), g(pre + "post_attention_layernorm.weight"))
        m = pre + "mlp."
        x = ref_rms(h, g(pre + "pre_feedforward_layernorm.weight"))
        x = lin(m + "down_proj.", ref_gelu_tanh(lin(m + "gate_proj.", x)) * lin(m + "up_proj.", x))
        h = h + ref_rms(x, g(pre + "post_feedforward_layernorm.weight"))
    out = ref_pool(h, pos, L // cfg.pooling_kernel_size ** 2)
    if cfg.standardize:
        out = (out - g("std_bias")) * g("std_scale")
    return out


def _vision_cfg(**kw):
    base = dict(hidden_size=16, intermediate_size=24, num_hidden_layers=2, num_attention_heads=2,
                num_key_value_heads=2, head_dim=8, patch_size=2, position_embedding_size=16,
                pooling_kernel_size=2, default_output_length=4,
                rope_parameters={"rope_theta": 100.0, "rope_type": "default"})
    base.update(kw)
    return VC.VisionConfig.from_dict(base)


def _vision_case(cfg, seed=1234):
    vm = V.VisionModel(cfg)
    rng = np.random.default_rng(seed)
    flat = []
    for k, v in tree_flatten(vm.parameters()):
        if k.endswith(("input_min", "output_min")):
            a = np.array(-0.8 if "input" in k else -1.5)
        elif k.endswith(("input_max", "output_max")):
            a = np.array(0.8 if "input" in k else 1.5)
        elif "norm" in k or "std_scale" in k:
            a = 1 + 0.3 * rng.standard_normal(v.shape)
        else:
            a = rng.standard_normal(v.shape) * 0.4
        flat.append((k, mx.array(a.astype(np.float32))))
    vm.update(tree_unflatten(flat))
    P = {k: _np(v) for k, v in tree_flatten(vm.parameters())}
    img = rng.random((3, 8, 8)).astype(np.float32)
    got = _np(vm(mx.array(img[None]))[0])
    return P, img, got


def test_vision_rope_2d():
    """HF axial 2D rope vs knurlogic apply_multidimensional_rope
    (vision.py:95-137), at e4b's head_dim 64 and 26B's 72 (not a multiple of 4
    halves cleanly: 2 * (72 // 4) = 36 per axis)."""
    rng = np.random.default_rng(1234)
    for D in (64, 72):
        x = rng.standard_normal((1, 6, 2, D)).astype(np.float32)
        pos = rng.integers(0, 40, (6, 2))
        got = _np(V.apply_multidimensional_rope(mx.array(x), mx.array(pos[None]), 100.0))[0]
        np.testing.assert_allclose(got, ref_vision_rope(x[0].astype(np.float64), pos, 100.0),
                                   rtol=1e-4, atol=1e-4, err_msg=str(D))


def test_vision_tower_e4b_like_matches_reference():
    """Whole tower on e4b's vision settings (use_clipped_linears, no
    standardize), clips set finite so they fire: knurlogic VisionModel
    (vision.py) vs ref_vision_forward."""
    cfg = _vision_cfg(use_clipped_linears=True, standardize=False)
    P, img, got = _vision_case(cfg)
    want = ref_vision_forward(cfg, P, img.astype(np.float64))
    np.testing.assert_allclose(got, want, rtol=2e-4, atol=2e-4)


def test_vision_tower_26b_like_matches_reference():
    """26B's vision settings: no clipped linears, standardize (std_bias/std_scale)."""
    cfg = _vision_cfg(use_clipped_linears=False, standardize=True)
    P, img, got = _vision_case(cfg)
    want = ref_vision_forward(cfg, P, img.astype(np.float64))
    np.testing.assert_allclose(got, want, rtol=2e-4, atol=2e-4)


def test_multimodal_embedder():
    """HF:2053-2077 Gemma4MultimodalEmbedder: unscaled RMSNorm then project,
    vs vision/__init__.py:51-65."""
    rng = np.random.default_rng(1234)
    m = VF.MultimodalEmbedder(16, 24)
    w = rng.standard_normal((24, 16)).astype(np.float32)
    m.embedding_projection.weight = mx.array(w)
    x = rng.standard_normal((5, 16)).astype(np.float32)
    np.testing.assert_allclose(_np(m(mx.array(x))), ref_rms(x.astype(np.float64)) @ w.T,
                               rtol=1e-4, atol=1e-5)


def test_image_feature_prescale_roundtrip():
    """The reference scatters image features UNSCALED into the (already
    sqrt(H)-scaled) text embeddings (HF:2306-2370); knurlogic's encode()
    divides them by sqrt(H) and the trunk multiplies everything back
    (vision/__init__.py:202-213, gemma4_text.py:576-580). Exact in fp32; in
    bf16 the round trip costs a rounding. Bound it (measured, not gated)."""
    rng = np.random.default_rng(1234)
    f = rng.standard_normal((64, 2560)).astype(np.float32) * 3
    s = 2560 ** 0.5
    for dt, tol in ((mx.float32, 1e-6), (mx.bfloat16, 1e-2)):
        a = mx.array(f).astype(dt)
        back = _np((a / s) * s)
        rel = np.abs(back - _np(a)).max() / np.abs(_np(a)).max()
        assert rel < tol, (dt, rel)


# ---------------------------------------------------------------- lab: real configs

def _cfgs():
    out = []
    for d in sorted(EXO.glob("*gemma-4*")) + sorted(HUB.glob("models--*gemma-4*/snapshots/*")):
        if (d / "config.json").exists():
            out.append((d, json.load(open(d / "config.json"))))
    return out


@pytest.mark.lab
def test_real_configs_read_as_reference():
    """On every gemma-4 config on this box (maker bf16 + ours): the runtime
    reads the fields the reference keys on -- layer_types, rope per type,
    final softcap, eps, KV sharing, k_eq_v / global KV heads, MoE, PLE; the
    vision tower's clipped linears / standardize; PLE's image placeholder id
    (knurlogic zeroes it, the reference uses pad_token_id)."""
    cs = _cfgs()
    if not cs:
        pytest.skip("no gemma-4 checkpoints on this machine")
    for d, c in cs:
        t = c["text_config"]
        a = W4.ModelArgs.from_dict(c)
        ta = A.ModelArgs.from_dict(a.text_config)
        for k in ("layer_types", "sliding_window", "final_logit_softcapping", "rms_norm_eps",
                  "num_kv_shared_layers", "attention_k_eq_v", "num_global_key_value_heads",
                  "enable_moe_block", "num_experts", "top_k_experts", "hidden_size_per_layer_input",
                  "global_head_dim", "head_dim", "use_double_wide_mlp", "tie_word_embeddings"):
            if k in t:
                assert getattr(ta, k) == t[k], (d.name, k, getattr(ta, k), t[k])
        for lt, rp in t["rope_parameters"].items():
            assert ta.rope_parameters[lt] == rp, (d.name, lt)
        assert t.get("pad_token_id", 0) == 0, d.name
        vc = VC.VisionConfig.from_dict(c["vision_config"])
        for k in ("use_clipped_linears", "standardize", "head_dim", "pooling_kernel_size",
                  "default_output_length", "patch_size", "rms_norm_eps"):
            if k in c["vision_config"]:
                assert getattr(vc, k) == c["vision_config"][k], (d.name, k)
        assert vc.rope_parameters["rope_theta"] == c["vision_config"]["rope_parameters"]["rope_theta"]


@pytest.mark.parametrize("kw", [E4B_LIKE, A26B_LIKE], ids=["e4b", "26b"])
def test_cached_decode_matches_reference(kw):
    """Prefill 5 tokens, then decode one at a time through make_cache()
    (RotatingKVCache on sliding layers, rope offsets, shared-KV layers reading
    the source layer's cache; gemma4_text.py:256-269, 601-625, 729-742): the
    logits of every position must equal the reference's full forward."""
    args = _args(**kw)
    m = A.Model(args)
    P = _randomize(m)
    ids = np.random.default_rng(99).integers(1, args.vocab_size, 11)
    want = ref_text_forward(args, P, ids)
    cache = m.make_cache()
    out = [_np(m(mx.array(ids[None, :5]), cache=cache)[0])]
    for t in range(5, len(ids)):
        out.append(_np(m(mx.array(ids[None, t:t + 1]), cache=cache)[0]))
    np.testing.assert_allclose(np.concatenate(out), want, rtol=2e-3, atol=2e-3)
