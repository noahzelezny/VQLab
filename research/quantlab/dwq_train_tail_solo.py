#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""E26: solo tail-DWQ — train tail layers + frozen norm/head on ONE box from
cached prefix activations (dwq_cache_student_prefix.py) and cached teacher
top-k logits (dwq_cache_teacher_stock.py).

Mathematically identical to full-model tail-only DWQ (the prefix is frozen,
so its activations are constants), but: no distribution, ~15G resident,
3.3s/step measured — the Metal-watchdog-proof formulation. Patches are full
unsharded tensors keyed by model path.
"""
import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optimizers
from mlx.utils import tree_flatten, tree_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--acts-dir", required=True)
    ap.add_argument("--target-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tail-layers", type=int, default=6)
    ap.add_argument("--learning-rate", type=float, default=1e-6)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--freeze-attention", action="store_true",
                    help="train ONLY mlp/expert scales — E27: trained "
                         "attention scales destroyed OOD behavior (layer-55 "
                         "activation inflation -> uniform logits on "
                         "wikitext) while acing on-distribution KL.")
    ap.add_argument("--ood-canary", default=None,
                    help="path to an OOD text file (wikitext-2 VALIDATION "
                         "slice — never the referee test corpus). Scored "
                         "via the frozen prefix is unavailable here, so the "
                         "canary uses cached canary acts (--canary-acts).")
    ap.add_argument("--canary-acts", default=None,
                    help="safetensors with {acts,tokens} for the canary "
                         "text's layer-cut activations + target tokens; "
                         "epoch gate refuses to save a patch whose canary "
                         "NLL regresses >0.5%% from the untrained tail.")
    ap.add_argument("--init-patch", default=None,
                    help="tail_patch.safetensors from a previous round — "
                         "continuation training starts from these trained "
                         "scales instead of the artifact's.")
    args = ap.parse_args()

    # Buffer-cache cap (E26 footprint fix #2): batches have VARIABLE seq
    # lengths, so each new length allocates a fresh ~25G transient set and
    # MLX's cache retains every size — measured 119G peak on the 96G M3
    # (sample(1) physical footprint; ps rss wildly undercounts unified
    # memory). Cap the cache so steady state stays ~model + optimizer.
    mx.set_cache_limit(8 << 30)

    from mlx_lm.utils import load

    with mx.stream(mx.cpu):
        model, _, _ = load(args.student, lazy=True, return_config=True)
    core = model
    for name in ("language_model", "model"):
        while hasattr(core, name):
            core = getattr(core, name)
    n_blocks = len(core.layers)
    cut = n_blocks - args.tail_layers
    tail = core.layers[cut:]
    head_norm = core.norm
    lm_head = getattr(model, "lm_head", None) or getattr(
        getattr(model, "language_model", model), "lm_head", None)
    with mx.stream(mx.cpu):
        for l in tail:
            mx.eval(l.parameters())
        mx.eval(head_norm.parameters(), lm_head.parameters())

    # SEVER the frozen prefix (E26 footprint fix): the lazy references to
    # 54 untrained layers keep ~107G of mmap'd safetensors reachable, and
    # touched pages accumulate until macOS memory pressure goes yellow and
    # every step pays compression tax. The loss path never touches them —
    # drop the references so the pages can't stay resident.
    import gc
    core.layers = [None] * cut + list(tail)
    core.embed_tokens = None
    gc.collect()
    mx.clear_cache()
    print(f"tail [{cut},{n_blocks}) + norm + head: "
          f"{mx.get_active_memory() / 1024**3:.1f}G", flush=True)

    def unfreeze(path, m):
        if args.freeze_attention and (
                "self_attn" in path or "linear_attn" in path):
            return
        if (hasattr(m, "bits") and hasattr(m, "group_size")
                and getattr(m, "mode", None) == "affine" and m.bits < 8):
            m.unfreeze(keys=["scales", "biases"], recurse=False)
    for l in tail:
        l.freeze()
        l.apply_to_modules(unfreeze)
        l.train()
    if args.freeze_attention:
        print("attention scales FROZEN (E27)", flush=True)
    head_norm.freeze()
    lm_head.freeze()

    if args.init_patch:
        from mlx.utils import tree_unflatten
        prev = mx.load(args.init_patch)
        by_layer = {}
        for k, v in prev.items():
            assert k.startswith("model.layers."), k
            rest = k[len("model.layers."):]
            li, sub = rest.split(".", 1)
            by_layer.setdefault(int(li), []).append((sub, v))
        for li, items in by_layer.items():
            assert cut <= li < n_blocks, f"patch layer {li} outside tail"
            tail[li - cut].update(tree_unflatten(items))
        print(f"resumed from {args.init_patch}: "
              f"{len(prev)} tensors into {len(by_layer)} layers", flush=True)

    params = {str(i): l.trainable_parameters() for i, l in enumerate(tail)}
    n = sum(v.size for _, v in tree_flatten(params))
    print(f"trainable {n / 1e6:.0f}M", flush=True)

    acts_dir, tgt_dir = Path(args.acts_dir), Path(args.target_dir)
    n_train = len(list((acts_dir / "train").glob("*.safetensors")))
    n_valid = len(list((acts_dir / "valid").glob("*.safetensors")))
    assert n_train == len(list((tgt_dir / "train").glob("*.safetensors"))), \
        "acts/targets train count mismatch — caches are misaligned"
    print(f"{n_train} train / {n_valid} valid batches", flush=True)

    scale = 1 / args.temperature

    def load_batch(split, i):
        a = mx.load(str(acts_dir / split / f"{i:010d}.safetensors"))
        t = mx.load(str(tgt_dir / split / f"{i:010d}.safetensors"))
        assert a["acts"].shape[1] == t["logits"].shape[1], \
            f"seq mismatch {split}#{i}: {a['acts'].shape} vs {t['logits'].shape}"
        return a["acts"], a["lengths"], t["logits"], t["indices"]

    from mlx_lm.tuner.losses import kl_div_loss

    def loss_fn(p, h, t_logits, t_idx, lengths):
        for i, l in enumerate(tail):
            l.update(tree_map(lambda x: x.astype(mx.bfloat16), p[str(i)]))
            h = l(h, mask=("causal" if not l.is_linear else None), cache=None)
        logits = lm_head(head_norm(h))
        logits = mx.take_along_axis(logits, t_idx, axis=-1)
        losses = kl_div_loss(scale * logits, scale * t_logits.astype(logits.dtype))
        mask = mx.arange(1, 1 + t_logits.shape[1]) < lengths[:, 1:]
        ntoks = mask.sum()
        return (mask * losses).sum() / ntoks, ntoks

    def validate(p):
        tot, toks = 0.0, 0
        for i in range(n_valid):
            h, ln, tl, ti = load_batch("valid", i)
            loss, nt = loss_fn(p, h, tl, ti, ln)
            mx.eval(loss, nt)
            tot += loss.item() * nt.item()
            toks += int(nt.item())
        return tot / max(toks, 1)

    canary = None
    if args.canary_acts:
        canary = mx.load(args.canary_acts)

    def canary_nll(p):
        """NLL of the OOD canary text through the (patched) tail + head."""
        h = canary["acts"]
        tgt = canary["tokens"][1: 1 + h.shape[1]]
        for i, l in enumerate(tail):
            l.update(tree_map(lambda x: x.astype(mx.bfloat16), p[str(i)]))
            h = l(h, mask=("causal" if not l.is_linear else None), cache=None)
        logits = lm_head(head_norm(h))[0].astype(mx.float32)
        lse = mx.logsumexp(logits, axis=-1)
        t = mx.take_along_axis(logits, tgt[:, None].astype(mx.int64),
                               axis=-1)[:, 0]
        out = (lse - t).mean()
        mx.eval(out)
        return float(out.item())

    fp32_params = tree_map(lambda x: x.astype(mx.float32), params)
    opt = optimizers.Adam(learning_rate=args.learning_rate,
                          bias_correction=True)
    grad_fn = mx.value_and_grad(loss_fn)

    v0 = validate(fp32_params)
    print(f"initial valid loss: {v0:.4f}", flush=True)
    c0 = canary_nll(fp32_params) if canary is not None else None
    if c0 is not None:
        print(f"initial canary NLL: {c0:.4f}", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    step = 0
    t0 = time.time()
    best = v0
    for ep in range(args.epochs):
        for i in range(n_train):
            if args.max_steps and step >= args.max_steps:
                break
            h, ln, tl, ti = load_batch("train", i)
            (loss, nt), grads = grad_fn(fp32_params, h, tl, ti, ln)
            fp32_params = opt.apply_gradients(grads, fp32_params)
            mx.eval(loss, fp32_params)
            if step == 0:
                gn = float(mx.sqrt(sum((g.astype(mx.float32) ** 2).sum()
                                       for _, g in tree_flatten(grads))).item())
                print(f"step-0 grad norm: {gn:.3e}", flush=True)
                assert gn > 1e-12, "zero gradients — wiring broken"
            step += 1
            if step % 20 == 0:
                print(f"ep{ep} step {step}/{args.epochs * n_train} "
                      f"loss={loss.item():.4f} "
                      f"({(time.time() - t0) / step:.1f}s/step)", flush=True)
                mx.clear_cache()
            if step % 100 == 0:
                ck = {}
                for i in range(len(tail)):
                    for k, val in tree_flatten(
                            tree_map(lambda x: x.astype(mx.bfloat16),
                                     fp32_params[str(i)])):
                        ck[f"model.layers.{cut + i}.{k}"] = val
                mx.save_safetensors(str(out / "checkpoint.safetensors"), ck)
        v = validate(fp32_params)
        print(f"epoch {ep} valid loss: {v:.4f} (initial {v0:.4f})", flush=True)
        if canary is not None:
            c = canary_nll(fp32_params)
            print(f"epoch {ep} canary NLL: {c:.4f} (initial {c0:.4f})",
                  flush=True)
            if c > c0 * 1.005:
                print("  CANARY REGRESSED — patch NOT saved (E27 gate)",
                      flush=True)
                continue
        if v < best:
            best = v
            patch = {}
            for i, l in enumerate(tail):
                for k, val in tree_flatten(
                        tree_map(lambda x: x.astype(mx.bfloat16),
                                 fp32_params[str(i)])):
                    patch[f"model.layers.{cut + i}.{k}"] = val
            mx.save_safetensors(str(out / "tail_patch.safetensors"), patch)
            (out / "meta.json").write_text(json.dumps({
                "student": args.student, "cut": cut,
                "tail_layers": args.tail_layers, "epoch": ep,
                "valid_loss_initial": v0, "valid_loss_best": best,
                "lr": args.learning_rate,
            }, indent=1))
            print(f"  saved patch ({len(patch)} tensors)", flush=True)
        elif v >= v0:
            print("  WARNING: no improvement over initial", flush=True)

    print(f"done: best valid {best:.4f} vs initial {v0:.4f}", flush=True)


if __name__ == "__main__":
    main()
