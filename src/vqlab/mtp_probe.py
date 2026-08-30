"""MTP acceptance probe (qwen4_exp): does the teacher's MTP head draft well
enough against a quantized main model to justify a decode loop?

Teacher-forced, greedy proxy for speculative acceptance: run the MAIN model
over the corpus once, capturing the pre-mixer hyper-connection row h_i and
the main model's own greedy prediction at every position. Then run the MTP
head at each position i with (h_i, embedding of the ACTUAL next token) and
ask how often its greedy draft for token i+2 matches what the main model
itself would produce there. That match rate upper-bounds greedy
speculative acceptance; ~chance means the head (or this wiring) is dead.

Head wiring per the llama.cpp qwen4-exp port (PR #27739, read 2026-08-30),
with the norm details settled by measurement (see report_norm_convention):
  h_row -> arch.RMSNorm(10240, group_size=2560)  one statistic per stream,
           matching every other wide norm in this arch (measured: grouped
           0.6992 vs flat-row 0.6562 acceptance)
        -> reshape to [hc, 2560] streams
  e     -> arch.RMSNorm(2560)         -> shared by every stream
  per stream: eh_proj([e | h_stream])  (fc_embedding|fc_hidden joined; both
           are separate [2560,2560] tensors, so concat order is NOT ambiguous)
  -> ONE standard qwen4_exp full-attention block (own 512-expert MoE)
  -> the head's own hyper_connection_mixer (carries the final norm)
  -> the shared lm_head / tied embedding.

    python -m vqlab.cli mtp-probe --model <artifact> --mtp <graft.safetensors>
        --corpus <txt> [--tokens N] [--out probe.json]
"""
import argparse
import json
import pathlib
import sys

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import runtime_load


def report_norm_convention(g):
    """Print the sidecar's RMSNorm gain statistics. Diagnostic only.

    Norm convention is what killed this probe: an MTP sidecar is loaded
    outside the trunk's sanitize, so whoever applies its norms owns the
    +1.0 zero-centered convention. The two architectures resolve it in
    OPPOSITE places, and conflating them costs a factor of nothing but
    produces exactly 0.0 acceptance either way:

      qwen3_5 (dense 27B): stored delta, applied by a conventional
        nn.RMSNorm. mlx-lm's trunk sanitize adds 1.0; a hand-loaded
        sidecar must be shifted the same way (0.0000 -> 0.7285 measured
        2026-08-30).
      qwen4_exp (this model): stored delta, applied by the arch's OWN
        zero-centered RMSNorm (y = norm(x) * (1 + weight)), which adds the
        1.0 itself. The stored weights must therefore be left ALONE and
        fed to arch.RMSNorm -- pre-shifting them here would double-count
        (0.0000 -> 0.6992 measured once the arch class was used).

    So this function deliberately does not mutate anything: the probe below
    uses the architecture's own norm classes, which is the only way the
    convention cannot drift.
    """
    means = {k: float(mx.mean(v.astype(mx.float32)).item())
             for k, v in g.items() if v.ndim == 1 and k.endswith("norm.weight")}
    if means:
        lo = min(means, key=means.get)
        hi = max(means, key=means.get)
        print(f"mtp norm gains: {len(means)} tensors, mean range "
              f"{means[lo]:+.3f} ({lo}) .. {means[hi]:+.3f} ({hi}); applied via "
              f"the architecture's own RMSNorm", flush=True)
    return g


def rms(x, w, eps=1e-6):
    x32 = x.astype(mx.float32)
    n = x32 * mx.rsqrt(mx.mean(mx.square(x32), axis=-1, keepdims=True) + eps)
    return (n * w.astype(mx.float32)).astype(x.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="main-model artifact dir")
    ap.add_argument("--mtp", required=True, help="bf16 mtp graft safetensors")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from mlx_lm.utils import load
    from mlx_lm.models.qwen4_exp import create_attention_mask, create_ssm_mask
    model, tok = load(a.model, lazy=False, trust_remote_code=True)
    print(runtime_load.resolved_runtime_note(model), flush=True)
    core = model.model

    ids = tok.encode(open(a.corpus).read())[: a.tokens + 2]
    bos = getattr(tok, "bos_token_id", None)
    if bos is not None and (not ids or ids[0] != bos):
        ids = [bos] + ids[: a.tokens + 1]
    S = len(ids) - 2                      # positions with an i+2 target
    inp = mx.array([ids[:-1]])            # main model sees tokens 0..S

    # ---- main pass: capture the pre-mixer hyper-connection row ----------
    h = core.embed_tokens(inp)
    mask = create_attention_mask(h, None)
    conv_mask = create_ssm_mask(h, None)
    prev_ctx = None
    if core.ple_layers:
        ctx = core.args.ngram_size - 1
        eos = core.args.eos_token_id
        eos = eos[0] if isinstance(eos, list) else eos
        prev_ctx = mx.full((1, ctx), eos, inp.dtype)
    h = mx.tile(h, (1, 1, core.hc))
    for blk in core.layers:
        h = blk(h, core.rope, mask, conv_mask, None, None, inp, prev_ctx)
    mx.eval(h)                            # [1, S+1, hc*D] pre-mixer row

    mixer = core.hyper_connection_mixer
    head = model.lm_head if not model.args.text.tie_word_embeddings else None
    def to_logits(row):
        out = mixer(row)
        return (head(out) if head is not None
                else core.embed_tokens.as_linear(out)).astype(mx.float32)
    main_pred = mx.argmax(to_logits(h), axis=-1)[0]      # [S+1]
    mx.eval(main_pred)
    ctrl = float(mx.mean((main_pred[:S] == mx.array(ids[1 : S + 1])
                          ).astype(mx.float32)).item())
    print(f"control: main greedy vs corpus next-token agreement {ctrl:.3f} "
          f"(sanity: should be ~0.4-0.7; ~0 means the capture is broken)",
          flush=True)

    # ---- build the MTP block from the graft -----------------------------
    g = report_norm_convention(
        {k[len("mtp."):]: v for k, v in mx.load(a.mtp).items()})
    import importlib
    arch = importlib.import_module(type(core).__module__)
    args_t = core.args
    mtp_theta = getattr(args_t, "mtp", {}) or {}
    fa_idx = [i for i, l in enumerate(core.layers)
              if l.layer_type == "full_attention"][0]
    blk_cls = type(core.layers[fa_idx])
    # ctor signature is (args, layer_idx); layer_type comes from
    # args.layer_types[idx]. The PLE submodule it may build stays
    # zero-initialized (strict=False below) and contributes nothing.
    mblk = blk_cls(args_t, fa_idx)
    mblk.ple = None      # the MTP head has no PLE bank; a zero-filled stand-in
                         # is not a no-op through this class, so remove it
    # graft keys are upstream-format: apply the same expert mapping the
    # model's sanitize applies to main layers (gate_up split, switch_mlp
    # rename), then FAIL LOUDLY on any key that finds no parameter slot.
    layer_w = {}
    for k, v in g.items():
        if not k.startswith("layers.0."):
            continue
        k = k[len("layers.0."):]
        if k.endswith("mlp.experts.gate_up_proj"):
            mid = v.shape[-2] // 2
            layer_w["mlp.switch_mlp.gate_proj.weight"] = v[..., :mid, :]
            layer_w["mlp.switch_mlp.up_proj.weight"] = v[..., mid:, :]
        elif k.endswith("mlp.experts.down_proj"):
            layer_w["mlp.switch_mlp.down_proj.weight"] = v
        else:
            layer_w[k] = v
    from mlx.utils import tree_flatten
    slots = {k for k, _ in tree_flatten(mblk.parameters())}
    unmatched = sorted(set(layer_w) - slots)
    if unmatched:
        raise SystemExit(f"FAIL: {len(unmatched)} graft keys found no "
                         f"parameter slot, e.g. {unmatched[:4]}")
    missing = sorted(slots - set(layer_w))
    if missing:
        print(f"note: {len(missing)} module params not in graft (left init), "
              f"e.g. {missing[:4]}", flush=True)
    mblk.load_weights(list(layer_w.items()), strict=False)
    mmixer_cls = type(mixer)
    mmixer = mmixer_cls(args_t, use_combine=False)
    mmixer.load_weights([(k[len("hyper_connection_mixer."):], v)
                         for k, v in g.items()
                         if k.startswith("hyper_connection_mixer.")],
                        strict=False)
    rope_theta = float(mtp_theta.get("rope_theta", 10_000_000))
    mtp_rope = type(core.rope)(core.rope.dim, rope_theta)
    mx.eval(mblk.parameters(), mmixer.parameters())

    D = args_t.hidden_size
    hc = core.hc

    # ---- MTP forward, sweeping the wiring ambiguities -------------------
    nxt = mx.array([ids[1:]])                              # [1, S+1]
    e_raw = core.embed_tokens(nxt)
    B, T, _ = h.shape

    # qwen4_exp's RMSNorm is ZERO-CENTERED (y = norm(x) * (1 + weight)) and the
    # 10240-wide norms take per-stream statistics (group_size=hidden_size).
    # Hand-rolling `n * w` here silently drops the +1 and drove acceptance to
    # 0.0; use the architecture's own class so the convention can't drift.
    eps = getattr(args_t, "rms_norm_eps", 1e-6)

    def arch_norm(dim, w, group_size=None):
        n = arch.RMSNorm(dim, group_size=group_size, eps=eps)
        n.weight = w.astype(mx.float32)
        return n

    ne = arch_norm(D, g["pre_fc_norm_embedding.weight"])
    e = ne(e_raw)
    e_s = mx.broadcast_to(e[:, :, None, :], (B, T, hc, D))
    W = mx.concatenate([g["fc_embedding.weight"], g["fc_hidden.weight"]], axis=1)

    m_mask = create_attention_mask(h[..., :D], None)
    results = {}
    # The one genuine remaining ambiguity: whether the wide hidden norm takes
    # one statistic per stream (as every other 10240-wide norm in this arch
    # does) or one over the flat row. Concat order is NOT ambiguous -- the head
    # carries separate fc_embedding / fc_hidden tensors.
    for norm_style, gs in (("group", D), ("row", None)):
        nh = arch_norm(hc * D, g["pre_fc_norm_hidden.weight"], group_size=gs)
        h_s = nh(h).reshape(B, T, hc, D)
        cat = mx.concatenate([e_s, h_s], axis=-1)
        hin = (cat @ W.T.astype(cat.dtype)).reshape(B, T, hc * D)
        hout = mblk(hin, mtp_rope, m_mask, None, None, None, nxt, prev_ctx)
        lg = (head(mmixer(hout)) if head is not None
              else core.embed_tokens.as_linear(mmixer(hout))
              ).astype(mx.float32)
        pred = mx.argmax(lg, axis=-1)[0]
        mx.eval(pred)
        acc = float(mx.mean((pred[:S] == main_pred[1 : S + 1])
                            .astype(mx.float32)).item())
        accc = float(mx.mean((pred[:S] == mx.array(ids[2 : S + 2]))
                             .astype(mx.float32)).item())
        results[norm_style] = (acc, accc)
        print(f"variant {norm_style}-norm: acc_vs_main {acc:.4f}  "
              f"vs_corpus {accc:.4f}  "
              f"sample: {tok.decode([int(x) for x in pred[:8].tolist()])!r}",
              flush=True)
    best = max(results.items(), key=lambda kv: kv[1][0])
    acc_main, acc_corpus = best[1]
    mtp_pred = None

    # ---- score ----------------------------------------------------------
    # MTP at position i drafts token i+2; the main model's own greedy
    # choice for i+2 is main_pred[i+1]. Also report agreement with the
    # actual corpus token as a secondary line.
    rec = {"model": a.model, "mtp": a.mtp, "corpus": a.corpus, "positions": S,
           "best_variant": best[0],
           "acceptance_vs_main_greedy": round(acc_main, 4),
           "agreement_vs_corpus": round(acc_corpus, 4)}
    print(json.dumps(rec), flush=True)
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
