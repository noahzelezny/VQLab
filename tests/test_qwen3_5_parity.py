"""qwen3_5 / qwen3_5_moe reference parity, CPU stream only, tiny shapes.

The reference is transformers 5.18 models/qwen3_5/modeling_qwen3_5.py and
models/qwen3_5_moe/modeling_qwen3_5_moe.py (abbreviated Q35 / Q35M below;
torch is not installed, so each behaviour is transcribed in float64 numpy
with its file:line). The runtime is the architecture `import vqlab` serves
as mlx_lm.models.qwen3_5 / qwen3_5_moe (knurlogic), which pulls attention,
MLP, gated norm and MoE block from mlx_lm.models.qwen3_next and the
recurrence from mlx_lm.models.gated_delta. PARITY in vqlab/family/qwen3_5.py
names these tests.

Every tiny model is built from a weight dict in the MAKER's checkpoint
layout (zero-centred norms, conv1d [C, 1, K], fused experts.gate_up_proj)
and loaded through the runtime's own sanitize(), so the load-time
conventions are exercised, not assumed.

CPU: everything runs under `with mx.stream(mx.cpu)`, which also routes
gated_delta_update to its ops path (gated_delta.py: the Metal kernel is
taken only when the default device is the GPU). The Metal kernel itself is
NOT exercised here (see the parity item gdn_metal_kernel)."""
import copy
import importlib
import json
import pathlib
import struct

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("knurlogic")
import vqlab  # noqa: E402,F401  serves knurlogic's qwen3_5 as mlx_lm.models.qwen3_5

A = importlib.import_module("mlx_lm.models.qwen3_5")
AM = importlib.import_module("mlx_lm.models.qwen3_5_moe")

MAKER = {
    "qwen3_5": pathlib.Path("/Volumes/Storage SSD/Exo Models/Qwen--Qwen3.5-2B"),
    "qwen3_5_moe": pathlib.Path("/Volumes/Storage HDD/Teacher Models/Qwen--Qwen3.6-35B-A3B-bf16"),
}

EPS = 1e-6
RTOL = ATOL = 2e-4


# --------------------------------------------------------------------------
# numpy reference (float64)
# --------------------------------------------------------------------------

def _silu(x):
    return x / (1.0 + np.exp(-x))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softplus(x):
    return np.logaddexp(x, 0.0)


def ref_rmsnorm(x, w, eps):
    """Q35:840-854 Qwen3_5RMSNorm: x * rsqrt(mean(x^2) + eps) * (1 + w)."""
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * (1.0 + w)


def ref_gated_rmsnorm(x, z, w, eps):
    """Q35:217-233 Qwen3_5RMSNormGated: norm BEFORE gate, plain weight
    (initialised to ones, NOT 1+w), times silu(z)."""
    return w * (x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)) * _silu(z)


def ref_l2norm(x, eps=1e-6):
    """Q35:293-296 l2norm: x * rsqrt(sum(x^2) + eps), eps=1e-6 (Q35:343, :463)."""
    return x / np.sqrt((x * x).sum(-1, keepdims=True) + eps)


def ref_causal_conv_silu(x, w):
    """Q35:268-288 causal_conv1d_fn: depthwise conv, left pad K-1, keep the
    first S outputs, then silu (activation = hidden_act, Q35:516).
    x [B, S, C], w [C, 1, K] (checkpoint layout)."""
    B, S, C = x.shape
    K = w.shape[-1]
    xp = np.concatenate([np.zeros((B, K - 1, C)), x], axis=1)
    out = sum(xp[:, j:j + S, :] * w[:, 0, j] for j in range(K))
    return _silu(out)


def ref_gdn(x, p, cfg):
    """Q35:548-662 Qwen3_5GatedDeltaNet.forward (no cache), recurrence as
    torch_recurrent_gated_delta_rule Q35:437-497 (the chunked form
    Q35:300-434 is the same recurrence reorganised)."""
    B, S, _ = x.shape
    Hk, Hv = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"]
    Dk, Dv = cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]
    kd = Hk * Dk
    qkv = ref_causal_conv_silu(x @ p["in_proj_qkv.weight"].T, p["conv1d.weight"])
    q = qkv[..., :kd].reshape(B, S, Hk, Dk)
    k = qkv[..., kd:2 * kd].reshape(B, S, Hk, Dk)
    v = qkv[..., 2 * kd:].reshape(B, S, Hv, Dv)
    z = (x @ p["in_proj_z.weight"].T).reshape(B, S, Hv, Dv)
    beta = _sigmoid(x @ p["in_proj_b.weight"].T)                       # Q35:616
    g = -np.exp(p["A_log"]) * _softplus(x @ p["in_proj_a.weight"].T + p["dt_bias"])  # Q35:618
    r = Hv // Hk
    q, k = np.repeat(q, r, axis=2), np.repeat(k, r, axis=2)            # repeat_interleave Q35:619-621
    q, k = ref_l2norm(q), ref_l2norm(k)                                # Q35:462-464
    q = q / np.sqrt(Dk)                                                # Q35:467
    st = np.zeros((B, Hv, Dk, Dv))
    out = np.zeros((B, S, Hv, Dv))
    for t in range(S):
        st = st * np.exp(g[:, t])[..., None, None]
        kv = (st * k[:, t, :, :, None]).sum(-2)
        delta = (v[:, t] - kv) * beta[:, t, :, None]
        st = st + k[:, t, :, :, None] * delta[:, :, None, :]
        out[:, t] = (st * q[:, t, :, :, None]).sum(-2)
    out = ref_gated_rmsnorm(out, z, p["norm.weight"], cfg["rms_norm_eps"])
    return out.reshape(B, S, -1) @ p["out_proj.weight"].T


def _rotate_half(x):
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], -1)


def ref_rope_cos_sin(pos3, cfg):
    """Q35:162-212: inv_freq over dim = head_dim * partial_rotary_factor,
    per-axis freqs recomposed by mrope_section (H at i%3==1, W at i%3==2,
    below 3*section), then cat(freqs, freqs). pos3 [3, B, L]."""
    rp = cfg["rope_parameters"]
    dim = int(cfg["head_dim"] * rp["partial_rotary_factor"])
    inv = 1.0 / (rp["rope_theta"] ** (np.arange(0, dim, 2) / dim))
    f = pos3[..., None].astype(np.float64) * inv                      # [3, B, L, dim/2]
    thw = f[0].copy()
    for ax, off in ((1, 1), (2, 2)):
        idx = slice(off, rp["mrope_section"][ax] * 3, 3)
        thw[..., idx] = f[ax][..., idx]
    emb = np.concatenate([thw, thw], -1)
    return np.cos(emb), np.sin(emb)


def ref_apply_rope(x, cos, sin):
    """Q35:673-708 apply_rotary_pos_emb: partial -- rotate the first
    cos.shape[-1] dims (rotate_half pairing), pass the rest. x [B, H, L, D]."""
    rd = cos.shape[-1]
    c, s = cos[:, None], sin[:, None]
    xr = x[..., :rd] * c + _rotate_half(x[..., :rd]) * s
    return np.concatenate([xr, x[..., rd:]], -1)


def ref_attention(x, p, cfg, pos3=None):
    """Q35:748-820 Qwen3_5Attention.forward, causal, eager softmax (Q35:723-745)."""
    B, L, _ = x.shape
    H, Hkv, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    qg = (x @ p["q_proj.weight"].T).reshape(B, L, H, 2 * hd)         # Q35:785-788
    q, gate = qg[..., :hd], qg[..., hd:].reshape(B, L, -1)
    q = ref_rmsnorm(q, p["q_norm.weight"], cfg["rms_norm_eps"]).transpose(0, 2, 1, 3)
    k = ref_rmsnorm((x @ p["k_proj.weight"].T).reshape(B, L, Hkv, hd), p["k_norm.weight"],
                    cfg["rms_norm_eps"]).transpose(0, 2, 1, 3)
    v = (x @ p["v_proj.weight"].T).reshape(B, L, Hkv, hd).transpose(0, 2, 1, 3)
    if pos3 is None:
        pos3 = np.broadcast_to(np.arange(L), (3, B, L))
    cos, sin = ref_rope_cos_sin(pos3, cfg)
    q, k = ref_apply_rope(q, cos, sin), ref_apply_rope(k, cos, sin)
    r = H // Hkv
    k, v = np.repeat(k, r, axis=1), np.repeat(v, r, axis=1)           # repeat_kv
    s = q @ k.transpose(0, 1, 3, 2) * hd ** -0.5                      # Q35:757
    s = s + np.triu(np.full((L, L), -np.inf), 1)
    s = np.exp(s - s.max(-1, keepdims=True))
    o = (s / s.sum(-1, keepdims=True)) @ v
    o = o.transpose(0, 2, 1, 3).reshape(B, L, -1) * _sigmoid(gate)    # Q35:816-817
    return o @ p["o_proj.weight"].T


def ref_mlp(x, p):
    """Q35:823-836: down(silu(gate(x)) * up(x)), hidden_act = silu."""
    return (_silu(x @ p["gate_proj.weight"].T) * (x @ p["up_proj.weight"].T)) @ p["down_proj.weight"].T


def ref_moe(x, p, cfg):
    """Q35M:844-923: softmax router (fp32), top-k, ALWAYS renormalised
    (Q35M:897, no norm_topk_prob switch); experts.gate_up_proj [E, 2I, D]
    chunked gate-first (Q35M:874); shared expert scaled by
    sigmoid(shared_expert_gate(x)) (Q35M:918)."""
    sh = x.shape
    x = x.reshape(-1, sh[-1])
    logits = x @ p["gate.weight"].T
    pr = np.exp(logits - logits.max(-1, keepdims=True))
    pr = pr / pr.sum(-1, keepdims=True)
    k = cfg["num_experts_per_tok"]
    idx = np.argsort(-pr, -1)[:, :k]
    w = np.take_along_axis(pr, idx, -1)
    w = w / w.sum(-1, keepdims=True)
    gu, dn = p["experts.gate_up_proj"], p["experts.down_proj"]
    I = gu.shape[1] // 2
    y = np.zeros_like(x)
    for t in range(x.shape[0]):
        for j in range(k):
            e = idx[t, j]
            h = gu[e] @ x[t]
            y[t] += w[t, j] * (dn[e] @ (_silu(h[:I]) * h[I:]))
    sp = {n: p["shared_expert." + n] for n in ("gate_proj.weight", "up_proj.weight", "down_proj.weight")}
    y = y + _sigmoid(x @ p["shared_expert_gate.weight"].T) * ref_mlp(x, sp)
    return y.reshape(sh)


def ref_forward(ids, W, cfg):
    """Q35:860-913 decoder layer (pre-norm, two residuals), Q35:1221-1300
    text model (final norm), lm_head tied to embed_tokens when
    tie_word_embeddings (Q35M: same)."""
    pre = "model.language_model."
    h = W[pre + "embed_tokens.weight"][ids]
    eps = cfg["rms_norm_eps"]
    for i, lt in enumerate(cfg["layer_types"]):
        lp = pre + f"layers.{i}."
        sub = lambda s: {k[len(lp + s):]: v for k, v in W.items() if k.startswith(lp + s)}  # noqa: E731
        hn = ref_rmsnorm(h, W[lp + "input_layernorm.weight"], eps)
        h = h + (ref_gdn(hn, sub("linear_attn."), cfg) if lt == "linear_attention"
                 else ref_attention(hn, sub("self_attn."), cfg))
        hn = ref_rmsnorm(h, W[lp + "post_attention_layernorm.weight"], eps)
        h = h + (ref_moe(hn, sub("mlp."), cfg) if cfg.get("num_experts") else ref_mlp(hn, sub("mlp.")))
    h = ref_rmsnorm(h, W[pre + "norm.weight"], eps)
    head = W[pre + "embed_tokens.weight"] if cfg["tie_word_embeddings"] else W["lm_head.weight"]
    return h @ head.T


# --------------------------------------------------------------------------
# tiny models in the maker's layout, loaded through the runtime's sanitize
# --------------------------------------------------------------------------

TEXT = {
    "hidden_size": 32, "num_hidden_layers": 2, "full_attention_interval": 2,
    "layer_types": ["linear_attention", "full_attention"],
    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
    "linear_num_key_heads": 2, "linear_num_value_heads": 4,
    "linear_key_head_dim": 8, "linear_value_head_dim": 8, "linear_conv_kernel_dim": 4,
    "intermediate_size": 24, "vocab_size": 50, "rms_norm_eps": 1e-6, "hidden_act": "silu",
    "max_position_embeddings": 4096, "attn_output_gate": True,
    "rope_parameters": {"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.5,
                        "mrope_section": [2, 1, 1], "mrope_interleaved": True},
}


def _cfg(moe: bool) -> dict:
    t = copy.deepcopy(TEXT)
    if moe:
        t.update(model_type="qwen3_5_moe_text", num_experts=6, num_experts_per_tok=2,
                 moe_intermediate_size=12, shared_expert_intermediate_size=10,
                 tie_word_embeddings=False)
    else:
        t.update(model_type="qwen3_5_text", tie_word_embeddings=True)
    return t


def _weights(t: dict, seed=1234) -> dict:
    """Random float64 weights under the checkpoint's key names and shapes."""
    rng = np.random.default_rng(seed)
    D = t["hidden_size"]
    n = lambda *s, sc=None: rng.standard_normal(s) * (sc if sc else s[-1] ** -0.5)  # noqa: E731
    pre = "model.language_model."
    W = {pre + "embed_tokens.weight": n(t["vocab_size"], D, sc=1.0),
         pre + "norm.weight": n(D, sc=0.2)}
    if not t["tie_word_embeddings"]:
        W["lm_head.weight"] = n(t["vocab_size"], D)
    Hk, Hv, Dk, Dv = (t["linear_num_key_heads"], t["linear_num_value_heads"],
                      t["linear_key_head_dim"], t["linear_value_head_dim"])
    C = 2 * Hk * Dk + Hv * Dv
    H, Hkv, hd = t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"]
    for i, lt in enumerate(t["layer_types"]):
        lp = pre + f"layers.{i}."
        W[lp + "input_layernorm.weight"] = n(D, sc=0.2)        # zero-centred (1 + w)
        W[lp + "post_attention_layernorm.weight"] = n(D, sc=0.2)
        if lt == "linear_attention":
            a = lp + "linear_attn."
            W[a + "in_proj_qkv.weight"] = n(C, D)
            W[a + "in_proj_z.weight"] = n(Hv * Dv, D)
            W[a + "in_proj_b.weight"] = n(Hv, D)
            W[a + "in_proj_a.weight"] = n(Hv, D)
            W[a + "conv1d.weight"] = n(C, 1, t["linear_conv_kernel_dim"], sc=0.5)
            W[a + "A_log"] = np.log(rng.uniform(0.5, 4.0, Hv))
            W[a + "dt_bias"] = n(Hv, sc=0.5)
            W[a + "norm.weight"] = 1.0 + n(Dv, sc=0.2)        # plain weight (ones-centred)
            W[a + "out_proj.weight"] = n(D, Hv * Dv)
        else:
            a = lp + "self_attn."
            W[a + "q_proj.weight"] = n(H * hd * 2, D)
            W[a + "k_proj.weight"] = n(Hkv * hd, D)
            W[a + "v_proj.weight"] = n(Hkv * hd, D)
            W[a + "o_proj.weight"] = n(D, H * hd)
            W[a + "q_norm.weight"] = n(hd, sc=0.2)
            W[a + "k_norm.weight"] = n(hd, sc=0.2)
        m = lp + "mlp."
        if t.get("num_experts"):
            E, I, Is = t["num_experts"], t["moe_intermediate_size"], t["shared_expert_intermediate_size"]
            W[m + "gate.weight"] = n(E, D, sc=1.0)
            W[m + "experts.gate_up_proj"] = n(E, 2 * I, D)
            W[m + "experts.down_proj"] = n(E, D, I)
            W[m + "shared_expert.gate_proj.weight"] = n(Is, D)
            W[m + "shared_expert.up_proj.weight"] = n(Is, D)
            W[m + "shared_expert.down_proj.weight"] = n(D, Is)
            W[m + "shared_expert_gate.weight"] = n(1, D)
        else:
            I = t["intermediate_size"]
            W[m + "gate_proj.weight"] = n(I, D)
            W[m + "up_proj.weight"] = n(I, D)
            W[m + "down_proj.weight"] = n(D, I)
    W["mtp.fc.weight"] = n(D, 2 * D)                            # sanitize must drop it
    return W


def _build(moe: bool, W=None):
    t = _cfg(moe)
    W = _weights(t) if W is None else W
    mod = AM if moe else A
    cfg = {"model_type": "qwen3_5_moe" if moe else "qwen3_5", "text_config": t}
    with mx.stream(mx.cpu):
        model = mod.Model(mod.ModelArgs.from_dict(cfg))
        sw = model.sanitize({k: mx.array(v.astype(np.float32)) for k, v in W.items()})
        model.load_weights(list(sw.items()), strict=True)
        model.eval()
        mx.eval(model.parameters())
    return model, W, t, sw


def _np(a):
    with mx.stream(mx.cpu):
        mx.eval(a)
    return np.array(a.astype(mx.float32)).astype(np.float64)


def _sub(W, prefix):
    return {k[len(prefix):]: v for k, v in W.items() if k.startswith(prefix)}


def _x(shape, seed=7, scale=1.0):
    return np.random.default_rng(seed).standard_normal(shape) * scale


L0 = "model.language_model.layers.0."
L1 = "model.language_model.layers.1."


# --------------------------------------------------------------------------
# the parity items
# --------------------------------------------------------------------------

def test_rmsnorm_zero_centred():
    """(1 + w) RMSNorm at the config eps: sanitize adds exactly 1.0 to every
    zero-centred norm (input/post-attention, final, q/k norm) and to nothing
    else -- not the gated norm, whose reference weight is plain."""
    model, W, t, sw = _build(False)
    lm = "language_model.model."
    for hf, rt in (("model.language_model.norm.weight", lm + "norm.weight"),
                   (L0 + "input_layernorm.weight", lm + "layers.0.input_layernorm.weight"),
                   (L1 + "post_attention_layernorm.weight", lm + "layers.1.post_attention_layernorm.weight"),
                   (L1 + "self_attn.q_norm.weight", lm + "layers.1.self_attn.q_norm.weight"),
                   (L1 + "self_attn.k_norm.weight", lm + "layers.1.self_attn.k_norm.weight")):
        np.testing.assert_allclose(_np(sw[rt]), W[hf] + 1.0, rtol=0, atol=1e-6)
    np.testing.assert_allclose(_np(sw[lm + "layers.0.linear_attn.norm.weight"]),
                               W[L0 + "linear_attn.norm.weight"], rtol=0, atol=1e-6)
    assert not any("mtp" in k for k in sw)
    x = _x((2, 5, t["hidden_size"]), scale=3.0)
    lay = model.language_model.model.layers[0]
    assert lay.input_layernorm.eps == t["rms_norm_eps"]
    with mx.stream(mx.cpu):
        got = _np(lay.input_layernorm(mx.array(x.astype(np.float32))))
    np.testing.assert_allclose(got, ref_rmsnorm(x, W[L0 + "input_layernorm.weight"], EPS),
                               rtol=RTOL, atol=ATOL)


def test_gated_rmsnorm():
    """RMSNormGated: plain weight, norm before gate, silu(z), config eps."""
    model, W, t, _ = _build(False)
    gn = model.language_model.model.layers[0].linear_attn.norm
    assert gn.eps == t["rms_norm_eps"]
    x, z = _x((2, 3, 4, 8), 1, 2.0), _x((2, 3, 4, 8), 2, 2.0)
    with mx.stream(mx.cpu):
        got = _np(gn(mx.array(x.astype(np.float32)), mx.array(z.astype(np.float32))))
    np.testing.assert_allclose(got, ref_gated_rmsnorm(x, z, W[L0 + "linear_attn.norm.weight"], EPS),
                               rtol=RTOL, atol=ATOL)


def test_gdn_causal_conv():
    """Depthwise causal conv (left pad K-1) + silu on the qkv projection,
    from the checkpoint's [C, 1, K] weight after sanitize's moveaxis."""
    model, W, t, _ = _build(False)
    g = model.language_model.model.layers[0].linear_attn
    x = _x((2, 6, g.conv_dim))
    with mx.stream(mx.cpu):
        xm = mx.array(x.astype(np.float32))
        zeros = mx.zeros((2, g.conv_kernel_size - 1, g.conv_dim))
        c = g.conv1d(mx.concatenate([zeros, xm], 1))
        got = _np(c * mx.sigmoid(c))
    np.testing.assert_allclose(got, ref_causal_conv_silu(x, W[L0 + "linear_attn.conv1d.weight"]),
                               rtol=RTOL, atol=ATOL)


def test_gdn_forward():
    """The whole GatedDeltaNet at O(1) activations: projections, conv, the
    decay g = -exp(A_log) * softplus(a + dt_bias), beta = sigmoid(b), q/k
    head repeat (interleaved), the Dk^-0.5 read-out scale, the delta-rule
    recurrence, gated norm and out_proj. At these magnitudes |k|^2 ~ O(1),
    so the l2norm eps (see test_gdn_l2norm_eps) is below tolerance here."""
    model, W, t, _ = _build(False)
    g = model.language_model.model.layers[0].linear_attn
    x = _x((2, 7, t["hidden_size"]), scale=2.0)
    with mx.stream(mx.cpu):
        got = _np(g(mx.array(x.astype(np.float32))))
    np.testing.assert_allclose(got, ref_gdn(x, _sub(W, L0 + "linear_attn."), t), rtol=RTOL, atol=ATOL)


def test_gdn_l2norm_eps():
    """q/k l2norm: reference x * rsqrt(sum(x^2) + 1e-6) (Q35:293-296,
    :343-344). The runtime writes it as Dk^-0.5 * rms_norm(x, eps=1e-6) =
    x * rsqrt(sum(x^2) + Dk * 1e-6): an effective eps Dk times too large
    (128x at the shipped Dk=128). Probed with small q/k (qkv projection
    scaled down) so the eps is visible. EXPECTED TO FAIL on knurlogic
    6a2ab49; the family-audit fix makes it pass."""
    t = _cfg(False)
    W = _weights(t)
    W[L0 + "linear_attn.in_proj_qkv.weight"] = W[L0 + "linear_attn.in_proj_qkv.weight"] * 1e-3
    model, W, t, _ = _build(False, W)
    g = model.language_model.model.layers[0].linear_attn
    x = _x((1, 5, t["hidden_size"]))
    with mx.stream(mx.cpu):
        got = _np(g(mx.array(x.astype(np.float32))))
    np.testing.assert_allclose(got, ref_gdn(x, _sub(W, L0 + "linear_attn."), t), rtol=1e-3, atol=1e-4)


def test_full_attention():
    """Gated full attention: per-head [q | gate] split of q_proj, (1 + w)
    q/k norm, partial rotary (first head_dim * partial_rotary_factor dims,
    rotate_half pairing) at rope_parameters.rope_theta, GQA, scale
    head_dim^-0.5, sigmoid(gate) on the output before o_proj."""
    model, W, t, _ = _build(False)
    at = model.language_model.model.layers[1].self_attn
    assert at.rope.dims == int(t["head_dim"] * t["rope_parameters"]["partial_rotary_factor"])
    assert at.rope.base == t["rope_parameters"]["rope_theta"] and not at.rope.traditional
    x = _x((2, 6, t["hidden_size"]))
    with mx.stream(mx.cpu):
        got = _np(at(mx.array(x.astype(np.float32)), mask="causal"))
    np.testing.assert_allclose(got, ref_attention(x, _sub(W, L1 + "self_attn."), t), rtol=RTOL, atol=ATOL)


def test_mrope_sections():
    """Interleaved MRoPE with distinct (t, h, w) positions: frequency i
    takes axis h when i % 3 == 1 and i < 3 * section[1], axis w when
    i % 3 == 2 and i < 3 * section[2], else t (Q35:204-212)."""
    t = copy.deepcopy(TEXT)
    t["head_dim"] = 32
    t["rope_parameters"] = dict(t["rope_parameters"], partial_rotary_factor=0.75,
                                mrope_section=[5, 4, 3], rope_theta=10000)
    rd = int(32 * 0.75)
    rng = np.random.default_rng(3)
    x = rng.standard_normal((1, 2, 5, 32))
    pos = rng.integers(0, 50, (3, 1, 5))
    with mx.stream(mx.cpu):
        got = _np(A.apply_mrope(mx.array(x.astype(np.float32)), mx.array(pos), rd,
                                t["rope_parameters"]["rope_theta"], t["rope_parameters"]["mrope_section"]))
    cos, sin = ref_rope_cos_sin(pos, t)
    np.testing.assert_allclose(got, ref_apply_rope(x, cos, sin), rtol=RTOL, atol=ATOL)


def test_dense_mlp():
    model, W, t, _ = _build(False)
    mlp = model.language_model.model.layers[0].mlp
    x = _x((2, 4, t["hidden_size"]))
    with mx.stream(mx.cpu):
        got = _np(mlp(mx.array(x.astype(np.float32))))
    np.testing.assert_allclose(got, ref_mlp(x, _sub(W, L0 + "mlp.")), rtol=RTOL, atol=ATOL)


def test_moe_block():
    """Softmax router, top-k, renormalised top-k weights, fused
    gate_up_proj split gate-first by sanitize, shared expert under a
    sigmoid gate."""
    model, W, t, _ = _build(True)
    blk = model.language_model.model.layers[0].mlp
    assert blk.norm_topk_prob is True          # the reference renormalises unconditionally
    x = _x((2, 5, t["hidden_size"]))
    with mx.stream(mx.cpu):
        got = _np(blk(mx.array(x.astype(np.float32))))
    np.testing.assert_allclose(got, ref_moe(x, _sub(W, L0 + "mlp."), t), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("moe", [False, True])
def test_full_forward(moe):
    """End to end: embed, pre-norm decoder with two residuals, layer
    schedule, final (1 + w) norm, tied (dense) / untied (MoE) head."""
    model, W, t, _ = _build(moe)
    ids = np.array([[3, 17, 4, 41, 9, 0, 22]])
    with mx.stream(mx.cpu):
        got = _np(model(mx.array(ids)))
    np.testing.assert_allclose(got, ref_forward(ids, W, t), rtol=5e-4, atol=5e-4)


def test_cached_decode():
    """Prefill 3 tokens then decode one at a time through make_cache():
    conv state, recurrent state, KV cache and rope offset carried across
    steps reproduce the reference's full-sequence logits."""
    model, W, t, _ = _build(False)
    ids = np.array([[5, 11, 2, 30, 8, 19]])
    ref = ref_forward(ids, W, t)
    with mx.stream(mx.cpu):
        cache = model.make_cache()
        outs = [_np(model(mx.array(ids[:, :3]), cache=cache))]
        for i in range(3, ids.shape[1]):
            outs.append(_np(model(mx.array(ids[:, i:i + 1]), cache=cache)))
    np.testing.assert_allclose(np.concatenate(outs, 1), ref, rtol=5e-4, atol=5e-4)


# --------------------------------------------------------------------------
# the maker checkpoints (read-only: config.json and safetensors headers)
# --------------------------------------------------------------------------

def _maker(mt):
    d = MAKER[mt]
    if not (d / "config.json").exists():
        pytest.skip(f"{mt} maker checkpoint not on this machine")
    return d


def _headers(d: pathlib.Path) -> dict:
    out = {}
    for f in sorted(set(json.load(open(d / "model.safetensors.index.json"))["weight_map"].values())):
        with open(d / f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            out.update(json.loads(fh.read(n)))
    out.pop("__metadata__", None)
    return out


@pytest.mark.lab
@pytest.mark.parametrize("mt", ["qwen3_5", "qwen3_5_moe"])
def test_maker_config_read(mt):
    """The runtime's reading of each maker config.json: layer schedule ==
    layer_types, rope theta / partial factor / mrope_section from
    rope_parameters, eps, tie, MoE top-k with renormalisation, and nothing
    the runtime ignores (attn_output_gate True, no mlp_only_layers)."""
    d = _maker(mt)
    cfg = json.load(open(d / "config.json"))
    assert cfg["model_type"] == mt
    t = cfg["text_config"]
    args = A.TextModelArgs.from_dict(copy.deepcopy(t))
    sched = ["linear_attention" if (i + 1) % args.full_attention_interval else "full_attention"
             for i in range(args.num_hidden_layers)]
    assert sched == t["layer_types"]
    rp = t["rope_parameters"]
    assert args.rope_theta == rp["rope_theta"]
    assert args.partial_rotary_factor == rp["partial_rotary_factor"]
    assert list(args.rope_parameters["mrope_section"]) == rp["mrope_section"]
    assert args.rope_parameters.get("type", "default") == "default"
    assert args.rms_norm_eps == t["rms_norm_eps"] and args.head_dim == t["head_dim"]
    assert args.tie_word_embeddings == t.get("tie_word_embeddings", False)
    assert t["hidden_act"] == "silu" and t.get("attn_output_gate", True) is True
    assert not t.get("mlp_only_layers")
    if mt == "qwen3_5_moe":
        assert args.num_experts == t["num_experts"] and args.num_experts_per_tok == t["num_experts_per_tok"]
        assert args.norm_topk_prob is True


@pytest.mark.lab
@pytest.mark.parametrize("mt", ["qwen3_5", "qwen3_5_moe"])
def test_maker_stored_layout(mt):
    """The stored tensors are in the layout sanitize converts from: conv1d
    [C, 1, K] (this is also the trigger for the +1.0 norm shift: a
    checkpoint whose conv is already [C, K, 1] gets NO shift), A_log
    present, MoE experts fused as gate_up_proj [E, 2I, D] / down_proj
    [E, D, I]."""
    d = _maker(mt)
    t = json.load(open(d / "config.json"))["text_config"]
    h = _headers(d)
    pre = "model.language_model.layers.0."
    K = t["linear_conv_kernel_dim"]
    C = 2 * t["linear_num_key_heads"] * t["linear_key_head_dim"] + \
        t["linear_num_value_heads"] * t["linear_value_head_dim"]
    assert h[pre + "linear_attn.conv1d.weight"]["shape"] == [C, 1, K]
    assert h[pre + "linear_attn.A_log"]["shape"] == [t["linear_num_value_heads"]]
    if mt == "qwen3_5_moe":
        E, I, D = t["num_experts"], t["moe_intermediate_size"], t["hidden_size"]
        assert h[pre + "mlp.experts.gate_up_proj"]["shape"] == [E, 2 * I, D]
        assert h[pre + "mlp.experts.down_proj"]["shape"] == [E, D, I]
