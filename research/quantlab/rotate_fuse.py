#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""Fused residual-stream Hadamard rotation (QuaRot/SpinQuant R1, weights-only).

Produces a rotated bf16 copy of a model that computes the SAME function as
the source (the R's cancel algebraically), but whose weight matrices have
outlier energy spread across channels — the property that makes 2-bit
affine grids fit better. Output is PLAIN WEIGHTS: stock mlx-lm loads it,
no runtime change (fused-only scope, Noah 2026-08-13).

Transform rules (MLX Linear layout W:[out, in], y = x W^T; residual x' = x H):
  residual READER  (in == hidden):  W' = (W * g_norm) @ H   (norm scale folded)
  residual WRITER  (out == hidden): W' = H^T @ W ; bias b' = b @ H
  embed_tokens (rows are residual vectors): E' = E @ H
  tied lm_head: untie — lm_head' = (E * g_final) @ H, embed' = E @ H.
  Untied lm_head: fold g_final, rotate as a reader.
  Every folded norm's weight -> ones. rms(xH) == rms(x) (orthogonal), so
  normalization itself commutes; only the learned scale doesn't — hence fold.

EVERY tensor with a hidden-sized axis must be explicitly classified;
anything unrecognized is a hard error, not a silent copy. The 2026-06
rotorquant salad (E31) is what an inconsistently-transformed model looks
like — the inventory is enforced in code and proven by the exactness test
(rotate WITHOUT quantizing -> PPL must match the source to float noise).
"""
import argparse
import json
import pathlib
import shutil

import numpy as np
import mlx.core as mx

mx.set_default_device(mx.cpu)


def hadamard(n):
    assert n & (n - 1) == 0, f"hidden {n} not a power of two"
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return mx.array((h / np.sqrt(n)).astype(np.float32))


READERS = (
    "linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
    "linear_attn.in_proj_a", "linear_attn.in_proj_b",
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "mlp.gate_proj", "mlp.up_proj",                      # dense MLP
    "mlp.switch_mlp.gate_proj", "mlp.switch_mlp.up_proj",  # MoE experts (mlx)
    "mlp.experts.gate_up_proj",                            # MoE experts (HF fused)
    "mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj",
    "mlp.gate", "mlp.shared_expert_gate",                 # router reads too
)
WRITERS = (
    "linear_attn.out_proj", "self_attn.o_proj",
    "mlp.down_proj", "mlp.switch_mlp.down_proj",
    "mlp.experts.down_proj",
    "mlp.shared_expert.down_proj",
)
KNOWN_SAFE = ("layernorm", ".norm.weight", "conv1d", "A_log", "dt_bias",
              "q_norm", "k_norm")


def norm_key_for(name):
    """RMSNorm weight whose output this reader consumes."""
    if "linear_attn" in name:
        return name.split("linear_attn")[0] + "input_layernorm.weight"
    if "self_attn" in name:
        return name.split("self_attn")[0] + "input_layernorm.weight"
    return name.split("mlp.")[0] + "post_attention_layernorm.weight"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard-gb", type=float, default=4.5)
    ap.add_argument("--sign-seed", type=int, default=None,
                    help="randomized Hadamard: compose a random ±1 diagonal "
                         "(this seed) with H. Still orthogonal, still fused. "
                         "Plain H (default) is the known-worst family member "
                         "— E32 measured it HURTING 2-bit affine quant.")
    args = ap.parse_args()
    src, out = pathlib.Path(args.src), pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.load(open(src / "config.json"))
    tc = cfg.get("text_config", cfg)
    hidden = tc["hidden_size"]
    tied = tc.get("tie_word_embeddings", cfg.get("tie_word_embeddings", False))
    H = hadamard(hidden)
    if args.sign_seed is not None:
        d = np.random.default_rng(args.sign_seed).choice(
            [-1.0, 1.0], size=hidden).astype(np.float32)
        H = mx.array(d[:, None]) * H  # R = D·H, orthogonal
        print(f"randomized Hadamard, sign seed {args.sign_seed}")

    idx_path = src / "model.safetensors.index.json"
    if idx_path.exists():
        wmap = json.load(open(idx_path))["weight_map"]
        files = sorted(set(wmap.values()))
    else:
        files = ["model.safetensors"]

    # pass 1: norm scales (small, needed to fold into readers anywhere).
    # HF-form checkpoints (mtp present / unsanitized conv1d) store norms
    # ZERO-CENTERED (w-1); mlx_lm's sanitize adds +1.0 at load. We fold the
    # EFFECTIVE scale (g+1) and emit a fully-sanitized output (mtp dropped,
    # conv1d moved), which mlx_lm loads with NO shift. Missing this was the
    # rung-2 71M-PPL failure — the 0.8B passed only because mlx-community
    # exports are already sanitized.
    hf_shift = False
    norms = {}
    for f in files:
        for k, v in mx.load(str(src / f)).items():
            if k.startswith("mtp."):
                hf_shift = True
                continue
            if "conv1d.weight" in k and v.shape[-1] != 1:
                hf_shift = True
            if "visual" in k or "vision" in k:
                continue
            if "layernorm.weight" in k or (k.endswith("norm.weight")
                                           and "layers" not in k):
                norms[k] = v.astype(mx.float32)
    if hf_shift:
        norms = {k: v + 1.0 for k, v in norms.items()}
    print(f"hf-form checkpoint: {hf_shift} (norm scales "
          f"{'shifted +1' if hf_shift else 'used as stored'})")
    final_norm_key = next(k for k in norms
                          if k.endswith("model.norm.weight")
                          or k.endswith("language_model.norm.weight"))

    consumed = set()
    stats = {"reader": 0, "writer": 0, "copy": 0}
    pending, pending_bytes, shard_i, out_map = {}, 0, 0, {}

    def flush(force=False):
        nonlocal pending, pending_bytes, shard_i
        if not pending or (not force and pending_bytes < args.shard_gb * 2**30):
            return
        shard_i += 1
        fname = f"model-{shard_i:05d}.safetensors"
        mx.save_safetensors(str(out / fname), pending)
        for k in pending:
            out_map[k] = fname
        pending, pending_bytes = {}, 0

    def put(name, arr):
        nonlocal pending_bytes
        pending[name] = arr
        pending_bytes += arr.nbytes
        flush()

    def as_bf16(a):
        return a.astype(mx.bfloat16)

    for f in files:
        shard = mx.load(str(src / f))
        for name, t in shard.items():
            if name.startswith("mtp."):
                # multi-token-prediction head: mlx_lm sanitize drops it in
                # every artifact we build; keeping it UNROTATED would be the
                # E31 failure in miniature, so drop it here too.
                stats["mtp_dropped"] = stats.get("mtp_dropped", 0) + 1
                continue
            base = name[:-len(".weight")] if name.endswith(".weight") else name

            if name.endswith("embed_tokens.weight"):
                E = t.astype(mx.float32)
                put(name, as_bf16(E @ H))
                if tied:
                    g = norms[final_norm_key]
                    put("lm_head.weight", as_bf16((E * g[None, :]) @ H))
                    consumed.add(final_norm_key)
                stats["reader"] += 1
                continue
            if base.endswith("lm_head") and not tied:
                g = norms[final_norm_key]
                put(name, as_bf16((t.astype(mx.float32) * g[None, :]) @ H))
                consumed.add(final_norm_key)
                stats["reader"] += 1
                continue
            if name in norms:
                continue  # emitted at the end

            role = None
            if any(base.endswith(s) for s in READERS):
                role = "reader"
            elif any(base.endswith(s) for s in WRITERS):
                role = "writer"

            if role == "reader":
                nk = norm_key_for(name)
                g = norms[nk]
                consumed.add(nk)
                W = t.astype(mx.float32)
                assert W.shape[-1] == hidden, (name, W.shape)
                gexp = g[None, None, :] if W.ndim == 3 else g[None, :]
                put(name, as_bf16((W * gexp) @ H))
                stats["reader"] += 1
                continue
            if role == "writer":
                W = t.astype(mx.float32)
                ax = 1 if W.ndim == 3 else 0
                assert W.shape[ax] == hidden, (name, W.shape)
                put(name, as_bf16(mx.swapaxes(
                    mx.swapaxes(W, ax, -1) @ H, ax, -1)))
                stats["writer"] += 1
                continue

            # vision->language merger writes the residual stream
            if "merger.linear_fc2" in name:
                if name.endswith(".weight") and t.shape[0] == hidden:
                    put(name, as_bf16(H.T @ t.astype(mx.float32)))
                    stats["writer"] += 1
                    continue
                if name.endswith(".bias") and t.shape[0] == hidden:
                    put(name, as_bf16(t.astype(mx.float32) @ H))
                    stats["writer"] += 1
                    continue

            if any(d == hidden for d in t.shape) and \
                    "vision_tower" not in name and ".visual." not in name:
                if not any(s in name for s in KNOWN_SAFE):
                    raise SystemExit(
                        f"UNCLASSIFIED hidden-sized tensor: {name} "
                        f"{t.shape} — extend the inventory, do not guess")
            if hf_shift and ("q_norm.weight" in name or "k_norm.weight" in name):
                t = (t.astype(mx.float32) + 1.0).astype(mx.bfloat16)
            elif hf_shift and "conv1d.weight" in name and t.shape[-1] != 1:
                t = mx.moveaxis(t, 2, 1)
            put(name, t)
            stats["copy"] += 1
        del shard

    for nk, g in norms.items():
        if nk in consumed:
            put(nk, as_bf16(mx.ones_like(g)))
        else:
            put(nk, as_bf16(g))
            stats["copy"] += 1
    flush(force=True)

    tsz = sum((out / f).stat().st_size for f in set(out_map.values()))
    json.dump({"metadata": {"total_size": tsz}, "weight_map": out_map},
              open(out / "model.safetensors.index.json", "w"))

    new_cfg = json.loads(json.dumps(cfg))
    for c in (new_cfg, new_cfg.get("text_config", {})):
        if "tie_word_embeddings" in c:
            c["tie_word_embeddings"] = False
    if tied:
        json.dump(new_cfg, open(out / "config.json", "w"), indent=1)
    else:
        shutil.copy2(src / "config.json", out / "config.json")

    for extra in ("tokenizer.json", "tokenizer_config.json", "vocab.json",
                  "chat_template.jinja", "generation_config.json",
                  "preprocessor_config.json", "processor_config.json",
                  "video_preprocessor_config.json", "merges.txt"):
        if (src / extra).exists():
            shutil.copy2(src / extra, out / extra)

    print(f"rotated {stats['reader']} readers + {stats['writer']} writers, "
          f"{len(consumed)} norms folded->ones, {stats['copy']} copied, "
          f"{tsz/2**30:.2f} GiB -> {out}")


if __name__ == "__main__":
    main()
