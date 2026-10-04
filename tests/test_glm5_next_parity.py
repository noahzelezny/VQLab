"""glm5_next (GLM-5.3-Flash) reference parity, CPU stream only, tiny shapes.

The reference is transformers 5.18's models/glm5_next/modeling_glm5_next.py
(cited below as M:<line>). torch is not installed here, so each behaviour is
transcribed into numpy and the knurlogic architecture module that VQLab loads
as mlx_lm.models.glm5_next is run against it on tiny random inputs. PARITY in
vqlab/family/glm5_next.py names these tests.

Weights go through the runtime's own LanguageModel.sanitize + load_weights
(checkpoint key names, conv/kv_b layouts), so the conversions are covered too.

Tests marked "known bug" FAILED on knurlogic 6a2ab49 (the failure message
carries the measured discrepancy) and pass on f03ec21, which fixed them. If
one fails again, fix the architecture, not the test.
"""
import json
import pathlib
import struct
import sys
import types

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("knurlogic")

# knurlogic's vendored mlx-vlm base imports PIL at module level for its image
# processor; no text path uses it. The vqlab venv has no Pillow, so stub it
# for the import only and take the stub out again (other tests importorskip PIL).
_STUB = False
try:
    import PIL  # noqa: F401
except ImportError:
    _STUB = True
    _pil = types.ModuleType("PIL")
    _img = types.ModuleType("PIL.Image")
    _img.Image = type("Image", (), {})
    _pil.Image = _img
    sys.modules["PIL"], sys.modules["PIL.Image"] = _pil, _img

import importlib  # noqa: E402

import vqlab  # noqa: E402,F401  serves knurlogic's architecture as mlx_lm.models.glm5_next

A = importlib.import_module("mlx_lm.models.glm5_next")
L = importlib.import_module("mlx_lm.models.glm5_next.language")
if _STUB:
    del sys.modules["PIL"], sys.modules["PIL.Image"]

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEACHER = pathlib.Path("/Volumes/Storage HDD/Teacher Models/zai-org--GLM-5.3-Flash-BF16")
LIMIT = 10.0
EPS = 1e-5          # the checkpoint's rms_norm_eps (config.json text_config)

# ----------------------------------------------------------------- fixtures

@pytest.fixture(autouse=True)
def _cpu_stream():
    """Every op in a test on the CPU stream, without touching the default
    device (mx.set_default_device leaks into later tests)."""
    with mx.stream(mx.cpu):
        yield


D, HC = 32, 4
TINY = dict(
    model_type="glm5_next_text", vocab_size=64, hidden_size=D, intermediate_size=16,
    moe_intermediate_size=8, num_hidden_layers=2, num_attention_heads=2,
    num_key_value_heads=2, n_shared_experts=1, n_routed_experts=8,
    routed_scaling_factor=2.5, kv_lora_rank=8, q_lora_rank=16, qk_rope_head_dim=0,
    v_head_dim=8, qk_nope_head_dim=8, qk_head_dim=8, num_experts_per_tok=2,
    first_k_dense_replace=1, max_position_embeddings=4096, rms_norm_eps=EPS,
    index_topk=8, index_head_dim=8, index_n_heads=4, index_kpool=4,
    layer_types=["linear_attention", "deepseek_sparse_attention"],
    mlp_layer_types=["dense", "sparse"], indexer_types=["full", "full"],
    linear_attn_config={"num_heads": 2, "head_dim": 8, "short_conv_kernel_size": 4,
                        "gate_lower_bound": -5.0},
    n_group=1, topk_group=1, norm_topk_prob=True, swiglu_limit=LIMIT,
    hc_mult=HC, hc_eps=1e-6, hc_sinkhorn_iters=20, mla_use_nope=True,
)


def _cfg(**over):
    return A.TextConfig.from_dict({**TINY, **over})


def _weights(cfg, rng, scale=0.3):
    """Checkpoint-named random weights (HF names, model.layers.<i>...)."""
    w = {}
    r = lambda *s: (rng.standard_normal(s) * scale).astype(np.float32)  # noqa: E731
    lin, H, hd = cfg.linear_num_heads, cfg.linear_num_heads, cfg.linear_head_dim
    qkv = lin * hd
    for i, lt in enumerate(cfg.layer_types):
        p = f"model.layers.{i}."
        for h in ("attn", "ffn"):
            w[p + f"hc_{h}_fn"] = r((2 + HC) * HC, HC * D)
            w[p + f"hc_{h}_base"] = r((2 + HC) * HC)
            w[p + f"hc_{h}_scale"] = 1 + r(3)
        w[p + "input_layernorm.weight"] = 1 + r(D)
        w[p + "post_attention_layernorm.weight"] = 1 + r(D)
        a = p + "self_attn."
        if lt == "linear_attention":
            for n in "qkv":
                w[a + f"{n}_proj.weight"] = r(qkv, D)
                w[a + f"{n}_conv1d.weight"] = r(qkv, 1, cfg.linear_conv_kernel_dim) * 3
            w[a + "f_a_proj.weight"] = r(hd, D)
            w[a + "f_b_proj.weight"] = r(qkv, hd)
            w[a + "dt_bias"] = r(qkv)
            w[a + "A_log"] = r(H)
            w[a + "b_proj.weight"] = r(H, D)
            w[a + "g_a_proj.weight"] = r(hd, D)
            w[a + "g_b_proj.weight"] = r(qkv, hd)
            w[a + "o_norm.weight"] = 1 + r(hd)
            w[a + "o_proj.weight"] = r(D, qkv)
        else:
            nh, nope, vd = cfg.num_attention_heads, cfg.qk_nope_head_dim, cfg.v_head_dim
            w[a + "q_a_proj.weight"] = r(cfg.q_lora_rank, D)
            w[a + "q_a_layernorm.weight"] = 1 + r(cfg.q_lora_rank)
            w[a + "q_b_proj.weight"] = r(nh * nope, cfg.q_lora_rank)
            w[a + "kv_a_proj_with_mqa.weight"] = r(cfg.kv_lora_rank, D)
            w[a + "kv_a_layernorm.weight"] = 1 + r(cfg.kv_lora_rank)
            w[a + "kv_b_proj.weight"] = r(nh * (nope + vd), cfg.kv_lora_rank)
            w[a + "o_proj.weight"] = r(D, nh * vd)
            ix, ih, ihd = a + "indexer.", cfg.index_n_heads, cfg.index_head_dim
            w[ix + "wq_b.weight"] = r(ih * ihd, cfg.q_lora_rank) * 3
            w[ix + "wk.weight"] = r(ihd, D) * 3
            w[ix + "k_norm.weight"] = 1 + r(ihd)
            w[ix + "k_norm.bias"] = r(ihd)
            w[ix + "weights_proj.weight"] = r(ih, D) * 3
            w[ix + "index_kpool_compress_ape"] = r(cfg.index_kpool, ihd)
            w[ix + "index_kpool_compress_gate"] = r(ihd, D)
        m = p + "mlp."
        if cfg.mlp_layer_types[i] == "dense":
            for n, s in (("gate_proj", (cfg.intermediate_size, D)), ("up_proj", (cfg.intermediate_size, D)),
                         ("down_proj", (D, cfg.intermediate_size))):
                w[m + n + ".weight"] = r(*s)
        else:
            I = cfg.moe_intermediate_size  # noqa: E741
            w[m + "gate.weight"] = r(cfg.n_routed_experts, D)
            w[m + "gate.e_score_correction_bias"] = r(cfg.n_routed_experts)
            for e in range(cfg.n_routed_experts):
                w[m + f"experts.{e}.gate_proj.weight"] = r(I, D)
                w[m + f"experts.{e}.up_proj.weight"] = r(I, D)
                w[m + f"experts.{e}.down_proj.weight"] = r(D, I)
            for n, s in (("gate_proj", (I, D)), ("up_proj", (I, D)), ("down_proj", (D, I))):
                w[m + "shared_experts." + n + ".weight"] = r(*s)
    w["model.embed_tokens.weight"] = r(cfg.vocab_size, D)
    w["model.norm.weight"] = 1 + r(D)
    w["lm_head.weight"] = r(cfg.vocab_size, D)
    return w


def _model(seed=1234, dtype=mx.float32, scale=0.3, **over):
    """(LanguageModel, checkpoint-named numpy weights) on the CPU stream."""
    cfg = _cfg(**over)
    w = _weights(cfg, np.random.default_rng(seed), scale)
    with mx.stream(mx.cpu):
        lm = L.LanguageModel(cfg)
        sw = lm.sanitize({k: mx.array(v) for k, v in w.items()})
        if dtype != mx.float32:                 # as the loader casts (cast_predicate)
            keep = lm.cast_predicate
            sw = {k: v.astype(dtype) if keep(k) else v for k, v in sw.items()}
        lm.load_weights(list(sw.items()), strict=True)
        mx.eval(lm.parameters())
    return lm, w


def _np(a):
    return np.array(a.astype(mx.float32))


def _bf16(a):
    """numpy fp32 -> bf16-rounded fp32 (round-to-nearest-even, as mlx casts)."""
    with mx.stream(mx.cpu):
        return np.array(mx.array(a).astype(mx.bfloat16).astype(mx.float32))


def _run(fn, *a):
    with mx.stream(mx.cpu):
        out = fn(*a)
        mx.eval(out)
    return out


# ------------------------------------------------------- numpy reference
# Everything below transcribes modeling_glm5_next.py (transformers 5.18).

def _silu(x):
    return x / (1.0 + np.exp(-x))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def ref_rmsnorm(x, w, eps):                         # M:67-81 Glm5NextTextRMSNorm
    x = x.astype(np.float32)
    return w * (x / np.sqrt((x * x).mean(-1, keepdims=True) + eps))


def ref_layernorm(x, w, b, eps):                    # nn.LayerNorm (indexer k_norm, M:801)
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def ref_swiglu(g, u, limit=LIMIT):                  # M:100-106, M:138-143
    return _silu(np.minimum(g, limit)) * np.clip(u, -limit, limit)


def ref_mlp(x, gw, uw, dw):                         # M:87-106 Glm5NextTextMLP
    return ref_swiglu(x @ gw.T, x @ uw.T) @ dw.T


def ref_router(x, W, bias, top_k, scale, norm=True, n_group=1, topk_group=1):
    """M:146-184 Glm5NextTextTopkRouter: fp32 logits, sigmoid, bias for the
    CHOICE only, group mask, top-k, weights from the unbiased scores."""
    logits = x.astype(np.float64) @ W.astype(np.float64).T          # fp32 in M:161
    s = _sigmoid(logits)
    choice = s + bias
    E = W.shape[0]
    gs = np.sort(choice.reshape(-1, n_group, E // n_group), -1)[..., -2:].sum(-1)
    keep = np.argsort(-gs, -1)[:, :topk_group]
    gmask = np.zeros_like(gs, bool)
    np.put_along_axis(gmask, keep, True, -1)
    choice = np.where(np.repeat(gmask, E // n_group, -1), choice, -np.inf)
    idx = np.argsort(-choice, -1, kind="stable")[:, :top_k]
    w = np.take_along_axis(s, idx, -1)
    if norm:
        w = w / (w.sum(-1, keepdims=True) + 1e-20)
    return idx, w * scale, logits


def ref_moe(x, w, p, cfg):                          # M:187-208 Glm5NextTextMoE
    idx, wt, _ = ref_router(x, w[p + "gate.weight"], w[p + "gate.e_score_correction_bias"],
                            cfg.num_experts_per_tok, cfg.routed_scaling_factor)
    out = np.zeros_like(x)
    for t in range(x.shape[0]):
        for j, e in enumerate(idx[t]):
            e = f"{p}experts.{e}."
            out[t] += wt[t, j] * ref_mlp(x[t:t + 1], w[e + "gate_proj.weight"], w[e + "up_proj.weight"],
                                         w[e + "down_proj.weight"])[0]
    s = p + "shared_experts."
    return out, out + ref_mlp(x, w[s + "gate_proj.weight"], w[s + "up_proj.weight"], w[s + "down_proj.weight"])


def ref_hc(streams, fn, base, scale, eps_norm, hc_eps, iters):
    """M:220-331 Glm5NextTextHyperConnection.forward -> (post, comb, collapsed)."""
    S, N, Dd = streams.shape
    f = streams.reshape(S, N * Dd).astype(np.float32)
    f = f / np.sqrt((f * f).mean(-1, keepdims=True) + eps_norm)     # M:211-217, eps=rms_norm_eps (M:278)
    mix = f @ fn.T
    pre_w, post_w, comb_w = mix[:, :N], mix[:, N:2 * N], mix[:, 2 * N:].reshape(S, N, N)
    pre = _sigmoid(pre_w * scale[0] + base[:N]) + hc_eps
    post = 2 * _sigmoid(post_w * scale[1] + base[N:2 * N])
    z = comb_w * scale[2] + base[2 * N:].reshape(N, N)
    z = np.exp(z - z.max(-1, keepdims=True))
    comb = z / z.sum(-1, keepdims=True) + hc_eps
    comb = comb / (comb.sum(-2, keepdims=True) + hc_eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdims=True) + hc_eps)
        comb = comb / (comb.sum(-2, keepdims=True) + hc_eps)
    collapsed = (pre[..., None] * streams).sum(1)
    return post, comb, collapsed


def ref_hc_expand(out, residual, post, comb):       # M:1337-1339 / M:1346-1348
    return post[..., None] * out[:, None, :] + np.matmul(np.swapaxes(comb, -1, -2), residual)


def ref_kda(x, w, a, cfg):
    """M:622-771 Glm5NextTextLinearAttention (prefill, no cache) with the
    M:465-516 recurrence (chunk_kimi_delta_attention M:519 computes the same
    function in chunks)."""
    S = x.shape[0]
    H, hd, K = cfg.linear_num_heads, cfg.linear_head_dim, cfg.linear_conv_kernel_dim
    mixed = np.concatenate([x @ w[a + f"{n}_proj.weight"].T for n in "qkv"], -1)       # [S, 3C]
    cw = np.concatenate([w[a + f"{n}_conv1d.weight"] for n in "qkv"], 0)[:, 0, :]     # [3C, K]
    pad = np.concatenate([np.zeros((K - 1, mixed.shape[1]), np.float32), mixed], 0)
    conv = np.stack([(pad[t:t + K] * cw.T).sum(0) for t in range(S)], 0)             # M:431-450
    conv = _silu(conv)
    C = H * hd
    q, k, v = (conv[:, i * C:(i + 1) * C].reshape(S, H, hd) for i in range(3))
    fg = (x @ w[a + "f_a_proj.weight"].T) @ w[a + "f_b_proj.weight"].T              # M:341-371
    g = (fg + w[a + "dt_bias"]).reshape(S, H, hd)
    g = cfg.linear_lower_bound * _sigmoid(np.exp(w[a + "A_log"])[None, :, None] * g)
    beta = _sigmoid(x @ w[a + "b_proj.weight"].T)                                    # M:735
    l2 = lambda t: t / np.sqrt((t * t).sum(-1, keepdims=True) + 1e-6)  # noqa: E731  M:453-462
    q, k = l2(q) * hd ** -0.5, l2(k)
    st = np.zeros((H, hd, hd), np.float64)                                           # [H, Dk, Dv]
    out = np.zeros((S, H, hd))
    for t in range(S):
        st = st * np.exp(g[t])[..., None]
        kv_mem = (st * k[t][..., None]).sum(-2)
        delta = (v[t] - kv_mem) * beta[t][:, None]
        st = st + k[t][..., None] * delta[:, None, :]
        out[t] = (st * q[t][..., None]).sum(-2)
    gate = ((x @ w[a + "g_a_proj.weight"].T) @ w[a + "g_b_proj.weight"].T).reshape(S, H, hd)
    o = ref_rmsnorm(out, w[a + "o_norm.weight"], cfg.rms_norm_eps) * _sigmoid(gate)  # M:376-395
    return o.reshape(S, -1) @ w[a + "o_proj.weight"].T


def ref_indexer(x, qr, w, a, cfg, bf16=False):
    """M:774-1062 Glm5NextTextIndexer (one sequence, no padding, no cache).
    Returns ({row: set(selected token indices)}, index_scores). bf16=True
    mirrors a bf16 run of the reference: Linear outputs and the pooled keys
    are bf16 (they are module outputs), the scoring from M:863 on is fp32."""
    rd = _bf16 if bf16 else (lambda t: t)
    S = x.shape[0]
    ix = a + "indexer."
    nH, hd, kp = cfg.index_n_heads, cfg.index_head_dim, cfg.index_kpool
    q = rd(qr @ w[ix + "wq_b.weight"].T).reshape(S, nH, hd)
    k = rd(ref_layernorm(rd(x @ w[ix + "wk.weight"].T), w[ix + "k_norm.weight"], w[ix + "k_norm.bias"], 1e-6))
    gate = rd(x @ w[ix + "index_kpool_compress_gate"].T)
    P = (S + kp - 1) // kp                                                           # M:937-1010
    pidx = np.arange(P * kp).reshape(P, kp)
    gv = pidx < S
    pool_valid = gv.all(-1)
    safe = np.clip(pidx, 0, S - 1)
    logits = gate[safe] + w[ix + "index_kpool_compress_ape"][None]
    logits = np.where(gv[..., None], logits, -np.inf)
    pr = np.exp(logits - logits.max(1, keepdims=True))
    pr = np.nan_to_num(pr / pr.sum(1, keepdims=True))
    pool_keys = rd((rd(pr) * k[safe]).sum(1))
    pidx, pool_keys = np.where(gv, pidx, -1)[pool_valid], pool_keys[pool_valid]      # keep, M:1008
    sc = np.maximum(np.einsum("shd,pd->shp", q, pool_keys) * hd ** -0.5, 0)          # M:863-864
    wts = rd(x @ w[ix + "weights_proj.weight"].T) * nH ** -0.5                       # M:867
    scores = np.einsum("sh,shp->sp", wts, sc)                                        # M:868
    pool_end = np.clip(pidx[:, -1], 0, S - 1)
    vis = pool_end[None, :] <= np.arange(S)[:, None]                                 # M:871-877, M:917-935
    scores = np.where(vis, scores, np.finfo(np.float32).min)
    sel_k = min(cfg.index_topk // kp, scores.shape[-1])
    out = {}
    for s in range(S):
        chosen = np.argsort(-scores[s], kind="stable")[:sel_k]
        toks = {int(t) for p in chosen if vis[s, p] for t in pidx[p] if t >= 0}
        cnt = s + 1                                                                  # M:1012-1062 tail
        tail = cnt % kp
        toks |= {cnt - tail + o for o in range(kp - 1) if o < tail}
        out[s] = toks
    return out, scores


def ref_mla(x, w, a, cfg, sel):
    """M:1102-1237 Glm5NextTextAttention, NoPE (qk_rope_head_dim = 0), with the
    indexer's selection as the mask (M:1239-1277) and eager softmax (M:1077)."""
    S = x.shape[0]
    H, nope, vd = cfg.num_attention_heads, cfg.qk_nope_head_dim, cfg.v_head_dim
    qr = ref_rmsnorm(x @ w[a + "q_a_proj.weight"].T, w[a + "q_a_layernorm.weight"], cfg.rms_norm_eps)
    q = (qr @ w[a + "q_b_proj.weight"].T).reshape(S, H, nope)
    kv = ref_rmsnorm(x @ w[a + "kv_a_proj_with_mqa.weight"].T, w[a + "kv_a_layernorm.weight"], cfg.rms_norm_eps)
    kvb = (kv @ w[a + "kv_b_proj.weight"].T).reshape(S, H, nope + vd)
    k, v = kvb[..., :nope], kvb[..., nope:]
    sc = np.einsum("shd,thd->hst", q, k) * (nope + 0) ** -0.5                        # M:1149 scaling
    allow = np.zeros((S, S), bool)
    for s, toks in sel.items():
        allow[s, list(toks)] = True
    sc = np.where(allow[None], sc, -np.inf)
    p = np.exp(sc - sc.max(-1, keepdims=True))
    p = p / p.sum(-1, keepdims=True)
    o = np.einsum("hst,thd->shd", p, v).reshape(S, -1)
    return o @ w[a + "o_proj.weight"].T, qr


# ----------------------------------------------------------- SwiGLU clamp

def _big(rng, shape, s=8.0):
    return (rng.standard_normal(shape) * s).astype(np.float32)


def test_dense_mlp_swiglu_clamp():
    """known bug: M:100-106 clamps gate <= 10 and |up| <= 10 in the dense MLP
    (layers 0-2); knurlogic's DeepseekMLP (mlp.py:29) applies plain swiglu."""
    lm, w = _model(scale=1.0)
    rng = np.random.default_rng(7)
    x = _big(rng, (6, D), 2.0)
    p = "model.layers.0.mlp."
    ref = ref_mlp(x, w[p + "gate_proj.weight"], w[p + "up_proj.weight"], w[p + "down_proj.weight"])
    pre_g = x @ w[p + "gate_proj.weight"].T
    pre_u = x @ w[p + "up_proj.weight"].T
    fired = float(((pre_g > LIMIT) | (np.abs(pre_u) > LIMIT)).mean())
    assert fired > 0, "fixture never exercised the clamp"
    got = _np(_run(lm.model.layers[0].mlp, mx.array(x)))
    err = np.abs(got - ref).max() / np.abs(ref).max()
    assert err < 1e-5, (f"dense MLP: no SwiGLU clamp at {LIMIT}: max |runtime - reference| / max|ref| "
                        f"= {err:.3g} with the clamp firing on {fired:.1%} of activations")


def test_shared_expert_swiglu_clamp():
    """known bug: the shared expert is a Glm5NextTextMLP (M:197-199), so it is
    clamped; knurlogic builds DeepseekV32MLP = DeepseekMLP, unclamped."""
    lm, w = _model(scale=1.0)
    x = _big(np.random.default_rng(8), (6, D), 2.0)
    p = "model.layers.1.mlp.shared_experts."
    ref = ref_mlp(x, w[p + "gate_proj.weight"], w[p + "up_proj.weight"], w[p + "down_proj.weight"])
    got = _np(_run(lm.model.layers[1].mlp.shared_experts, mx.array(x)))
    err = np.abs(got - ref).max() / np.abs(ref).max()
    assert err < 1e-5, f"shared expert: no SwiGLU clamp at {LIMIT}: max rel error {err:.3g}"


def test_routed_expert_swiglu_clamp():
    """known bug: Glm5NextTextExperts._apply_gate (M:138-143) clamps; knurlogic's
    SwitchGLU keeps the default SwiGLU() activation (switch_layers.py:162).
    The routed sum is isolated by zeroing the shared expert."""
    lm, w = _model(scale=1.0)
    p = "model.layers.1.mlp."
    for n in ("gate_proj", "up_proj", "down_proj"):
        w[p + f"shared_experts.{n}.weight"][:] = 0
    moe = lm.model.layers[1].mlp
    moe.shared_experts.down_proj.weight = mx.zeros_like(moe.shared_experts.down_proj.weight)
    x = _big(np.random.default_rng(9), (6, D), 2.0)
    routed, _ = ref_moe(x, w, p, lm.args)
    got = _np(_run(moe, mx.array(x)))
    err = np.abs(got - routed).max() / np.abs(routed).max()
    assert err < 1e-4, f"routed experts: no SwiGLU clamp at {LIMIT}: max rel error {err:.3g}"


# --------------------------------------------------------------- routing

def test_router_semantics_fp32():
    """M:146-184 in fp32: sigmoid scores, e_score_correction_bias for the
    choice only, n_group/topk_group = 1, norm_topk_prob, routed_scaling 2.5."""
    lm, w = _model()
    gate = lm.model.layers[1].mlp.gate
    rng = np.random.default_rng(10)
    x = rng.standard_normal((32, D)).astype(np.float32)
    W, b = w["model.layers.1.mlp.gate.weight"], w["model.layers.1.mlp.gate.e_score_correction_bias"] * 4
    gate.e_score_correction_bias = mx.array(b)
    inds, sc = _run(gate, mx.array(x))
    ri, rw, _ = ref_router(x, W, b, 2, 2.5)
    inds, sc = np.array(inds), _np(sc)
    assert (np.sort(inds, -1) == np.sort(ri, -1)).all()
    o, ro = np.argsort(inds, -1), np.argsort(ri, -1)
    np.testing.assert_allclose(np.take_along_axis(sc, o, -1), np.take_along_axis(rw, ro, -1), rtol=1e-5)
    plain = np.argsort(-(x @ W.T), -1)[:, :2]
    assert (np.sort(plain, -1) != np.sort(ri, -1)).any(), "bias never changed a pick"


def test_router_logits_fp32():
    """known bug: M:161 computes router logits as fp32(x) @ fp32(W);
    knurlogic's MoEGate (deepseek_v32/language.py:276) multiplies in the
    activation dtype and upcasts after (group_expert_select :236)."""
    lm, w = _model(dtype=mx.bfloat16, n_routed_experts=64, num_experts_per_tok=8)
    gate = lm.model.layers[1].mlp.gate
    rng = np.random.default_rng(11)
    x = _bf16(rng.standard_normal((256, D)).astype(np.float32) * 3)
    W = _np(gate.weight)
    b = _np(gate.e_score_correction_bias)
    inds, sc = _run(gate, mx.array(x).astype(mx.bfloat16))
    inds, sc = np.array(inds), _np(sc)
    ri, rw, _ = ref_router(x, W, b, 8, 2.5)
    flips = float((np.sort(inds, -1) != np.sort(ri, -1)).any(-1).mean())
    same = (np.sort(inds, -1) == np.sort(ri, -1)).all(-1)
    o, ro = np.argsort(inds, -1), np.argsort(ri, -1)
    dw = np.abs(np.take_along_axis(sc, o, -1) - np.take_along_axis(rw, ro, -1))[same].max()
    assert flips == 0 and dw < 1e-5, (
        f"router logits not fp32: {flips:.1%} of tokens pick a different expert set; "
        f"where the set agrees, max |weight - reference| = {dw:.3g}")


# ----------------------------------------------------------------- norms

def test_mla_latent_norm_eps():
    """known bug: q_a_layernorm / kv_a_layernorm use config.rms_norm_eps
    (1e-5, M:1129, M:1137); knurlogic hard-codes eps=1e-6 (language.py:524, :531)."""
    lm, w = _model()
    att = lm.model.layers[1].self_attn
    got = (att.q_a_layernorm.eps, att.kv_a_layernorm.eps)
    rng = np.random.default_rng(12)
    x = (rng.standard_normal((4, 16)) * 3e-3).astype(np.float32)   # small-RMS row: eps matters
    wq = w["model.layers.1.self_attn.q_a_layernorm.weight"]
    ref = ref_rmsnorm(x, wq, EPS)
    out = _np(_run(att.q_a_layernorm, mx.array(x)))
    err = np.abs(out - ref).max() / np.abs(ref).max()
    assert got == (EPS, EPS), (f"MLA latent norms eps {got}, reference {EPS}: on a row of RMS 3e-3 "
                               f"the q_a_layernorm output differs by {err:.3g} (relative)")


def test_indexer_k_norm_eps():
    """known bug: the indexer's k_norm is nn.LayerNorm(eps=1e-6) (M:801);
    knurlogic builds nn.LayerNorm(head_dim) = mlx's default eps 1e-5 (language.py:256)."""
    lm, w = _model()
    kn = lm.model.layers[1].self_attn.indexer.k_norm
    ix = "model.layers.1.self_attn.indexer."
    x = (np.random.default_rng(13).standard_normal((4, 8)) * 3e-3).astype(np.float32)
    ref = ref_layernorm(x, w[ix + "k_norm.weight"], w[ix + "k_norm.bias"], 1e-6)
    out = _np(_run(kn, mx.array(x)))
    err = np.abs(out - ref).max() / np.abs(ref).max()
    assert kn.eps == 1e-6, f"indexer k_norm eps {kn.eps}, reference 1e-6: relative error {err:.3g} at RMS 3e-3"


def test_block_norms_eps():
    """input/post-attention/final RMSNorm and the HC input norm use
    rms_norm_eps (M:1295-1296, M:1439, M:278); KDA's gated o_norm too (M:660)."""
    lm, _ = _model()
    m, eps = lm.model, lm.args.rms_norm_eps
    assert m.norm.eps == eps
    for lay in m.layers:
        assert lay.input_layernorm.eps == eps and lay.post_attention_layernorm.eps == eps
        assert lay.attn_hc.norm_eps == eps and lay.ffn_hc.norm_eps == eps
        assert lay.attn_hc.hc_eps == lm.args.hc_eps
    assert m.layers[0].self_attn.o_norm.eps == eps


# -------------------------------------------------------- hyper-connections

def test_hyper_connection_matches_reference():
    """M:220-331 (pre/post sigmoids, Sinkhorn comb, collapse) and the expand
    in the decoder layer (M:1337-1339), on the CPU ops path."""
    lm, w = _model()
    hc = lm.model.layers[0].attn_hc
    rng = np.random.default_rng(14)
    xs = rng.standard_normal((5, HC, D)).astype(np.float32)
    p = "model.layers.0.hc_attn_"
    rp, rc, rx = ref_hc(xs, w[p + "fn"], w[p + "base"], w[p + "scale"], EPS, 1e-6, 20)
    xc, post, comb = _run(hc, mx.array(xs[None]))
    np.testing.assert_allclose(_np(post)[0], rp, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(_np(comb)[0], rc, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(_np(xc)[0], rx, rtol=1e-5, atol=1e-5)
    out = rng.standard_normal((5, D)).astype(np.float32)
    got = _run(L.hc_expand, mx.array(out[None]), mx.array(xs[None]), post, comb)
    np.testing.assert_allclose(_np(got)[0], ref_hc_expand(out, xs, rp, rc), rtol=1e-4, atol=1e-5)


def test_hc_head_is_unweighted_mean():
    """M:334-338: the final HC collapse is an unweighted MEAN over streams,
    then model.norm (M:1511) -- not DeepSeek-V4's learned HyperHead."""
    lm, w = _model()
    m = lm.model
    mult = mx.array(np.array([1.0, 2.0, -3.0, 5.0], np.float32))[None, None, :, None]

    class Spread:
        is_linear = True

        def __call__(self, h, mask=None, cache=None):
            return h * mult + 0.25

    class Through:
        is_linear = False

        def __call__(self, h, mask=None, cache=None):
            return h

    m.layers = [Spread(), Through()]          # model.fa_idx / ssm_idx index these
    ids = np.array([[3, 9, 17]])
    got = _np(_run(lambda i: m(i, cache=[None, None]), mx.array(ids)))[0]
    e = w["model.embed_tokens.weight"][ids[0]]
    streams = e[:, None, :] * np.array([1, 2, -3, 5], np.float32)[None, :, None] + 0.25
    ref = ref_rmsnorm(streams.mean(1), w["model.norm.weight"], EPS)
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)


# ------------------------------------------------- KDA linear attention

def test_kda_linear_attention_matches_reference():
    """M:622-771 + M:341-371 + M:376-395 + M:431-462 + M:465-516: q/k/v
    projections, depthwise causal conv (K=4) + silu, fp32 l2-normed q/k,
    q * Dk^-0.5, safe forget gate lower_bound * sigmoid(exp(A_log) * g),
    beta = sigmoid(b), delta rule, gated o_norm -- with checkpoint conv
    weights [C, 1, K] through the runtime's sanitize."""
    lm, w = _model()
    x = np.random.default_rng(15).standard_normal((9, D)).astype(np.float32)
    ref = ref_kda(x, w, "model.layers.0.self_attn.", lm.args)
    got = _np(_run(lambda t: lm.model.layers[0].self_attn(t, None, None), mx.array(x[None])))[0]
    np.testing.assert_allclose(got, ref, rtol=2e-4, atol=2e-5)


# ------------------------------------------------ DSA indexer + NoPE MLA

S_DSA = 24


def _dsa_inputs(lm, w, seed=16):
    x = np.random.default_rng(seed).standard_normal((S_DSA, D)).astype(np.float32)
    _, qr = ref_mla(x, w, "model.layers.1.self_attn.", lm.args, {s: set(range(s + 1)) for s in range(S_DSA)})
    sel, scores = ref_indexer(x, qr, w, "model.layers.1.self_attn.", lm.args)
    return x, qr, sel, scores


def _sets(topk):
    t = np.array(topk)[0, 0]
    return {s: {int(i) for i in row if i >= 0} for s, row in enumerate(t)}


def test_indexer_selection_fp32():
    """M:774-1062 in fp32: k-pool compression (softmax over gate + ape),
    relu(q.k * hd^-0.5) weighted by weights_proj * nH^-0.5, a pool selectable
    only once its last token is visible, top (index_topk // kpool) pools,
    then the visible incomplete tail appended. The k_norm eps is set to the
    reference's here so this isolates the algorithm from that known bug."""
    lm, w = _model()
    att = lm.model.layers[1].self_attn
    att.indexer.k_norm.eps = 1e-6
    att.q_a_layernorm.eps = EPS
    x, qr, sel, scores = _dsa_inputs(lm, w)
    top = _run(att.indexer, mx.array(x[None]), att.q_a_layernorm(att.q_a_proj(mx.array(x[None]))), None)
    got = _sets(top)
    gaps = [np.sort(scores[s][scores[s] > -1e30])[::-1] for s in range(S_DSA)]
    tight = [s for s, gp in enumerate(gaps) if len(gp) > 2 and abs(gp[1] - gp[2]) < 1e-5]
    bad = [s for s in range(S_DSA) if got[s] != sel[s] and s not in tight]
    assert len(tight) < S_DSA // 4, f"{len(tight)} near-tied rows"
    assert not bad, f"rows {bad}: runtime {[sorted(got[s]) for s in bad]} reference {[sorted(sel[s]) for s in bad]}"
    assert any(len(sel[s]) < s + 1 for s in range(S_DSA)), "fixture never made a sparse selection"


def test_indexer_short_context_is_dense():
    """knurlogic skips the indexer while T <= index_topk (language.py:342) and
    attends densely. The reference selects every visible token there, so the
    two are the same set: checked on the reference."""
    lm, w = _model(index_topk=32)
    x = np.random.default_rng(17).standard_normal((S_DSA, D)).astype(np.float32)
    _, qr = ref_mla(x, w, "model.layers.1.self_attn.", lm.args, {s: set(range(s + 1)) for s in range(S_DSA)})
    sel, _ = ref_indexer(x, qr, w, "model.layers.1.self_attn.", lm.args)
    assert all(sel[s] == set(range(s + 1)) for s in range(S_DSA))
    att = lm.model.layers[1].self_attn
    assert _run(att.indexer, mx.array(x[None]), mx.array(qr[None]), None) is None


def test_indexer_fp32_scoring():
    """known bug: the reference scores pools in fp32 (q.float() @
    pool_keys.float(), M:863; weights .float(), M:867; pool softmax in fp32,
    M:1000-1004). knurlogic scores in the activation dtype (language.py:409-412,
    :286-290). Measured on a bf16 module against the reference run on the SAME
    bf16-rounded weights and inputs, rounding only where the reference's
    Linear outputs are bf16."""
    lm, w = _model(dtype=mx.bfloat16, index_n_heads=8, index_topk=8)
    att = lm.model.layers[1].self_attn
    att.indexer.k_norm.eps = 1e-6
    wb = {k: _bf16(v) for k, v in w.items()}
    n, rows = 0, 0
    for seed in range(6):
        x = _bf16(np.random.default_rng(100 + seed).standard_normal((48, D)).astype(np.float32))
        xb = mx.array(x[None]).astype(mx.bfloat16)
        qr_b = _run(lambda t: att.q_a_layernorm(att.q_a_proj(t)), xb)
        sel, _ = ref_indexer(x, _np(qr_b)[0], wb, "model.layers.1.self_attn.", lm.args, bf16=True)
        got = _sets(_run(att.indexer, xb, qr_b, None))
        rows += len(sel)
        n += sum(got[s] != sel[s] for s in sel)
    assert n == 0, (f"indexer not scored in fp32: {n}/{rows} query rows ({n / rows:.1%}) select a different "
                    f"token set from the fp32-scored reference on identical bf16 inputs")


def _mla_forward(att, x):
    return att(x, None, None)


def test_dsa_mla_prefill_matches_reference():
    """M:1102-1237: NoPE MLA (qk_rope_head_dim = 0, scale qk_head_dim^-0.5),
    kv_b_proj split k_nope | v per head (via the runtime's sanitize into
    embed_q / unembed_out), attention restricted to the indexer's selection.
    Expanded (prefill) path. Latent/k_norm eps set to the reference's to
    isolate this from those known bugs."""
    lm, w = _model()
    att = lm.model.layers[1].self_attn
    att.q_a_layernorm.eps = att.kv_a_layernorm.eps = EPS
    att.indexer.k_norm.eps = 1e-6
    x, qr, sel, _ = _dsa_inputs(lm, w)
    ref, _ = ref_mla(x, w, "model.layers.1.self_attn.", lm.args, sel)
    mask = L.create_attention_mask(mx.array(x[None]), None, return_array=True)
    got = _np(_run(lambda t: att(t, mask, None), mx.array(x[None])))[0]
    np.testing.assert_allclose(got, ref, rtol=2e-4, atol=2e-5)


def test_dsa_mla_decode_matches_reference():
    """The absorbed decode path (L <= SMALL_L: queries into the latent,
    gathered selection, language.py:569-612) with the incremental indexer
    cache: prefill 20 tokens, then 4 single-token steps, against the
    reference's full-sequence rows."""
    lm, w = _model()
    att = lm.model.layers[1].self_attn
    att.q_a_layernorm.eps = att.kv_a_layernorm.eps = EPS
    att.indexer.k_norm.eps = 1e-6
    x, qr, sel, _ = _dsa_inputs(lm, w)
    ref, _ = ref_mla(x, w, "model.layers.1.self_attn.", lm.args, sel)
    cache = lm.make_cache()[1]
    P0 = 20
    with mx.stream(mx.cpu):
        xa = mx.array(x[None])
        mask = L.create_attention_mask(xa[:, :P0], cache[0], return_array=True)
        outs = [att(xa[:, :P0], mask, cache)]
        for t in range(P0, S_DSA):
            outs.append(att(xa[:, t:t + 1], None, cache))
        got = _np(mx.concatenate(outs, axis=1))[0]
    np.testing.assert_allclose(got[P0:], ref[P0:], rtol=2e-4, atol=2e-5)


# ------------------------------------------------------- decoder wiring

def test_decoder_layer_wiring():
    """M:1280-1350: hc collapse -> input_layernorm -> mixer -> hc expand,
    then hc collapse -> post_attention_layernorm -> mlp -> hc expand, for a
    KDA + dense layer and a DSA + MoE layer. Small weights keep every SwiGLU
    input below the clamp, and the known-eps norms are set to the
    reference's, so this checks the wiring alone."""
    lm, w = _model(scale=0.05)
    cfg = lm.args
    rng = np.random.default_rng(18)
    h = (rng.standard_normal((S_DSA, HC, D))).astype(np.float32)
    att = lm.model.layers[1].self_attn
    att.q_a_layernorm.eps = att.kv_a_layernorm.eps = EPS
    att.indexer.k_norm.eps = 1e-6
    mask = L.create_attention_mask(mx.array(h[None]), None, return_array=True)
    for i, lay in enumerate(lm.model.layers):
        p = f"model.layers.{i}."
        post, comb, xc = ref_hc(h, w[p + "hc_attn_fn"], w[p + "hc_attn_base"], w[p + "hc_attn_scale"], EPS, 1e-6, 20)
        xn = ref_rmsnorm(xc, w[p + "input_layernorm.weight"], EPS)
        if lay.is_linear:
            r = ref_kda(xn, w, p + "self_attn.", cfg)
        else:
            _, qr = ref_mla(xn, w, p + "self_attn.", cfg, {s: set(range(s + 1)) for s in range(S_DSA)})
            sel, _ = ref_indexer(xn, qr, w, p + "self_attn.", cfg)
            r, _ = ref_mla(xn, w, p + "self_attn.", cfg, sel)
        h1 = ref_hc_expand(r, h, post, comb)
        post, comb, xc = ref_hc(h1, w[p + "hc_ffn_fn"], w[p + "hc_ffn_base"], w[p + "hc_ffn_scale"], EPS, 1e-6, 20)
        xn = ref_rmsnorm(xc, w[p + "post_attention_layernorm.weight"], EPS)
        if cfg.mlp_layer_types[i] == "dense":
            m = ref_mlp(xn, w[p + "mlp.gate_proj.weight"], w[p + "mlp.up_proj.weight"], w[p + "mlp.down_proj.weight"])
        else:
            m = ref_moe(xn, w, p + "mlp.", cfg)[1]
        ref = ref_hc_expand(m, h1, post, comb)
        got = _np(_run(lambda t: lay(t, mask=None if lay.is_linear else mask, cache=None), mx.array(h[None])))[0]
        np.testing.assert_allclose(got, ref, rtol=5e-4, atol=5e-5, err_msg=f"layer {i}")
        h = ref


# ------------------------------------------------- maker checkpoint (lab)

def _header(f):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n))


@pytest.mark.lab
def test_checkpoint_config_matches_runtime():
    """The maker checkpoint's text_config, read by the runtime's own
    TextConfig: every value these parity tests assume, and the stored
    dtypes the reference keeps in fp32 (M:1379 _keep_in_fp32_modules_strict)."""
    if not TEACHER.exists():
        pytest.skip("GLM-5.3-Flash teacher not on this machine")
    raw = json.load(open(TEACHER / "config.json"))
    assert raw["model_type"] == "glm5_next"
    t = raw["text_config"]
    cfg = A.TextConfig.from_dict(t)
    assert cfg.swiglu_limit == 10.0 and cfg.rms_norm_eps == 1e-5 and cfg.hc_eps == 1e-6
    assert (cfg.scoring_func, cfg.topk_method, cfg.n_group, cfg.topk_group) == ("sigmoid", "noaux_tc", 1, 1)
    assert cfg.norm_topk_prob and cfg.routed_scaling_factor == 2.5 and cfg.moe_router_dtype == "float32"
    assert cfg.qk_rope_head_dim == 0 and cfg.mla_use_nope                     # NoPE: no rope anywhere
    assert cfg.linear_lower_bound == -5.0 and cfg.linear_conv_kernel_dim == 4
    assert cfg.linear_num_heads == 64 and cfg.linear_head_dim == 128
    assert set(t["indexer_types"]) == {"full"}, "shared-indexer layers (M:1151) are not implemented by the runtime"
    assert (cfg.index_kpool, cfg.index_kpool_always_select_tail) == (4, True)
    # dense vs MoE layers: the runtime's rule (language.py:628-632) == mlp_layer_types
    rt = ["sparse" if i >= cfg.first_k_dense_replace and m == "sparse" else "dense"
          for i, m in enumerate(t["mlp_layer_types"])]
    assert rt == t["mlp_layer_types"]
    idx = json.load(open(TEACHER / "model.safetensors.index.json"))["weight_map"]
    pre = "model.language_model.layers."
    want = {pre + "3.mlp.gate.e_score_correction_bias": "F32", pre + "0.self_attn.A_log": "F32",
            pre + "0.self_attn.dt_bias": "F32", pre + "0.hc_attn_base": "F32", pre + "0.hc_attn_scale": "F32"}
    for k, dt in want.items():
        assert _header(TEACHER / idx[k])[k]["dtype"] == dt, k
