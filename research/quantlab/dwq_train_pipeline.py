#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""End-to-end DWQ for a model too big for one box: pipeline-parallel student,
teacher targets precomputed on disk (dwq_cache_teacher_stock.py).

Differences from mlx_lm.quant.dwq.dwq_quantize, each load-bearing:
  * The student is pipeline-sharded via the quantlab qwen3_5 PipelineMixin
    shim (E26) — each rank materializes ONLY its layer slice + embed/norm/
    lm_head, so a 122G student fits 96G+128G boxes.
  * NO nn.average_gradients: ranks hold DIFFERENT parameters in pipeline
    mode; averaging would mix unrelated tensors. Each rank steps its own
    Adam on its own slice.
  * GRADIENT-REACH CHECK on the first step: if mx.distributed send/recv do
    not propagate gradients, upstream layers would silently stop training.
    Every rank reports its grad norm; rank 0 aborts the run if any rank
    reports ~0. A silent no-op run is the failure mode this exists for.
  * Targets ALWAYS from --target-dir (top-1024 logits + indices per batch,
    stock format). No teacher anywhere in memory.

Launch (M3):
  ~/quantlab/venv/bin/mlx.launch --hosts 10.0.0.1,10.0.0.2 --backend ring \
      ~/quantlab/dwq_train_pipeline.py \
      --quantized-model <student> --target-dir <targets> --mlx-path <out>
"""
import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optimizers
import numpy as np
from mlx.utils import tree_flatten, tree_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantized-model", required=True)
    ap.add_argument("--target-dir", required=True)
    ap.add_argument("--mlx-path", required=True)
    ap.add_argument("--data-path", default="allenai/tulu-3-sft-mixture")
    ap.add_argument("--num-samples", type=int, default=2048)
    ap.add_argument("--max-seq-length", type=int, default=513)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--learning-rate", type=float, default=1e-6)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="stop after N train steps (0 = full pass); smoke "
                         "tests use small N")
    ap.add_argument("--tensor", action="store_true",
                    help="tensor-shard the student instead of pipeline. "
                         "REQUIRED for training: mlx send/recv have no vjp "
                         "(pipeline backward is impossible), while tensor "
                         "sharding's all_reduce is differentiable. Safe for "
                         "context-triggered artifacts at short seq (E23: "
                         "verify the artifact is context- not cumulative-"
                         "triggered FIRST via --reset-ctx x3 PASS).")
    ap.add_argument("--micro-batch", type=int, default=0,
                    help="split each cached batch into micro-batches of "
                         "this many rows, accumulating gradients before the "
                         "single optimizer step (semantics preserved). At "
                         "397B a full bs-4 fwd+bwd+update in one graph "
                         "exceeds the Metal watchdog on the 96G box — "
                         "smaller submissions are the fix. 0 = whole batch.")
    ap.add_argument("--train-last-layers", type=int, default=0,
                    help="unfreeze only the LAST N layers (0 = all). E25: "
                         "tail experts are where the value is, and it cuts "
                         "optimizer state ~20x — the difference between "
                         "fitting and not at 397B.")
    args = ap.parse_args()

    group = mx.distributed.init()
    rank, world = group.rank(), group.size()

    def rprint(*a):
        if rank == 0:
            print(*a, flush=True)

    rprint(f"world={world} — pipeline DWQ")

    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    from mlx_lm.quant.dwq import load_data
    from mlx_lm.tuner.losses import kl_div_loss
    from mlx_lm.tuner.trainer import grad_checkpoint, iterate_batches
    from mlx_lm.utils import load, save

    if args.tensor and world > 1:
        from mlx_lm.utils import sharded_load
        with mx.stream(mx.cpu):
            model, tokenizer, config = sharded_load(
                args.quantized_model, None, group, True)
        assert "quantization" in config, "student must be quantized"
        core = model
        for name in ("language_model", "model"):
            while hasattr(core, name):
                core = getattr(core, name)
        train_layers = list(core.layers)
        rprint(f"tensor-sharded: {len(train_layers)} layers, "
               f"1/{world} of each per rank")
    else:
        with mx.stream(mx.cpu):
            model, tokenizer, config = load(
                args.quantized_model, lazy=True, return_config=True)
        assert "quantization" in config, "student must be quantized"
        core = model
        for name in ("language_model", "model"):
            while hasattr(core, name):
                core = getattr(core, name)
        if world > 1:
            core.pipeline(group)
        train_layers = list(core.pipeline_layers)
        rprint(f"rank layer slices: local={len(train_layers)}")
        with mx.stream(mx.cpu):
            mx.eval(core.embed_tokens.parameters(), core.norm.parameters())
            head = getattr(model, "lm_head", None) or getattr(
                getattr(model, "language_model", model), "lm_head", None)
            if head is not None:
                mx.eval(head.parameters())
            for l in train_layers:
                mx.eval(l.parameters())
    print(f"[rank {rank}] materialized, "
          f"{mx.get_active_memory() / 1024**3:.1f}G active", flush=True)

    if args.train_last_layers:
        train_layers = train_layers[-args.train_last_layers:]
        rprint(f"training restricted to last {len(train_layers)} layers")

    train_data, valid_data = load_data(
        tokenizer, args.data_path, args.num_samples, args.max_seq_length)
    target_dir = Path(args.target_dir)

    def target_fn(idx, split):
        t = mx.load(str(target_dir / split / f"{idx:010d}.safetensors"))
        return t["logits"], t["indices"]

    # DWQ trainable set: affine-quantized scales/biases, bits < 8,
    # THIS RANK'S slice only (freeze() covers everything else incl. embed).
    def unfreeze(_, m):
        if (hasattr(m, "bits") and hasattr(m, "group_size")
                and getattr(m, "mode", None) == "affine" and m.bits < 8):
            m.unfreeze(keys=["scales", "biases"], recurse=False)

    model.freeze()
    for l in train_layers:
        l.apply_to_modules(unfreeze)
    model.train()

    n_train = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"[rank {rank}] trainable params: {n_train / 1e6:.1f}M", flush=True)

    # BARRIER before the first distributed forward. Without it, a rank that
    # finishes setup early enters recv() and blocks inside a Metal command
    # buffer while the other rank is still loading the HF dataset — the GPU
    # watchdog then kills it ("Ignored (for causing prior/excessive GPU
    # errors)"). Cost is one all_sum; it removes a whole class of
    # startup-skew crashes. On the CPU stream so waiting is free.
    ready = mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu)
    mx.eval(ready)
    assert int(ready.item()) == world, "barrier mismatch"
    rprint("all ranks ready (barrier passed)")

    if args.grad_checkpoint:
        grad_checkpoint(train_layers[0])

    if mx.metal.is_available():
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])

    scale = 1 / args.temperature
    dtype = mx.bfloat16

    def loss_fn(params, x, targets, lengths):
        model.update(tree_map(lambda p: p.astype(dtype), params))
        logits = model(x)
        t_logits, t_idx = targets
        logits = mx.take_along_axis(logits, t_idx, axis=-1)
        losses = kl_div_loss(scale * logits, scale * t_logits.astype(logits.dtype))
        mask = mx.arange(1, 1 + t_logits.shape[1]) < lengths[:, 1:]
        ntoks = mask.sum()
        return (mask * losses).sum() / ntoks, ntoks

    opt = optimizers.Adam(learning_rate=args.learning_rate,
                          bias_correction=True)
    params = tree_map(lambda p: p.astype(mx.float32),
                      model.trainable_parameters())

    def validate(params):
        v_loss, v_toks = 0.0, 0
        for i, (batch, lengths) in enumerate(iterate_batches(
                valid_data, args.batch_size, args.max_seq_length,
                seed=args.seed)):
            batch = mx.array(batch[:, :-1])
            loss, ntoks = loss_fn(params, batch, target_fn(i, "valid"),
                                  mx.array(lengths))
            mx.eval(loss, ntoks)
            v_loss += loss.item() * ntoks.item()
            v_toks += ntoks.item()
        return v_loss / max(v_toks, 1)

    v0 = validate(params)
    rprint(f"initial valid loss: {v0:.4f}")

    grad_fn = mx.value_and_grad(loss_fn)
    t0 = time.time()
    seen = 0
    for it, (batch, lengths) in enumerate(iterate_batches(
            train_data, args.batch_size, args.max_seq_length,
            seed=args.seed)):
        if args.max_steps and it >= args.max_steps:
            break
        batch = mx.array(batch[:, :-1])
        t_logits, t_idx = target_fn(it, "train")
        lengths = mx.array(lengths)
        rows = batch.shape[0]
        mb = args.micro_batch if args.micro_batch else rows
        grads = None
        loss_sum, ntoks_sum = 0.0, 0
        for s in range(0, rows, mb):
            e = min(s + mb, rows)
            (l, nt), g = grad_fn(params, batch[s:e],
                                 (t_logits[s:e], t_idx[s:e]), lengths[s:e])
            mx.eval(g)          # force each micro-batch's buffers separately
            loss_sum += l.item() * nt.item()
            ntoks_sum += int(nt.item())
            grads = g if grads is None else tree_map(mx.add, grads, g)
        if rows > mb:
            grads = tree_map(lambda x: x * (mb / rows), grads)
        params = opt.apply_gradients(grads, params)
        mx.eval(params)
        loss = mx.array(loss_sum / max(ntoks_sum, 1))
        ntoks = mx.array(ntoks_sum)

        if it == 0:
            # gradient-reach check — every rank must see nonzero grads
            gnorm = float(
                mx.sqrt(sum((g.astype(mx.float32) ** 2).sum()
                            for _, g in tree_flatten(grads))).item())
            print(f"[rank {rank}] step-0 grad norm: {gnorm:.3e}", flush=True)
            ok = mx.distributed.all_sum(
                mx.array(1.0 if gnorm > 1e-12 else 0.0), stream=mx.cpu)
            mx.eval(ok)
            if int(ok.item()) != world:
                if rank == 0:
                    print("FATAL: a rank reports zero gradients — "
                          "send/recv is not propagating grads; pipeline "
                          "training would silently no-op. Aborting.",
                          flush=True)
                raise SystemExit(1)
            rprint("gradient-reach check PASSED on all ranks")

        seen += int(ntoks.item())
        if rank == 0 and (it + 1) % 20 == 0:
            print(f"it={it + 1} loss={loss.item():.4f} "
                  f"tok/s={seen / (time.time() - t0):.0f} "
                  f"peak={mx.get_peak_memory() / 1024**3:.1f}G", flush=True)

    v1 = validate(params)
    rprint(f"final valid loss: {v1:.4f} (initial {v0:.4f})")
    if v1 >= v0:
        rprint("WARNING: no improvement — do not ship this artifact.")

    model.update(tree_map(lambda p: p.astype(dtype), params))

    # Each rank saves its OWN slice's patch; rank 0 also writes run meta.
    out = Path(args.mlx_path)
    out.mkdir(parents=True, exist_ok=True)
    patch = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(out / f"patch_rank{rank}.safetensors"), patch)
    if rank == 0:
        (out / "meta.json").write_text(json.dumps({
            "student": args.quantized_model, "target_dir": str(target_dir),
            "valid_loss_initial": v0, "valid_loss_final": v1,
            "num_samples": args.num_samples, "seed": args.seed,
            "learning_rate": args.learning_rate, "world": world,
        }, indent=1))
    print(f"[rank {rank}] saved patch ({len(patch)} tensors) -> {out}",
          flush=True)


if __name__ == "__main__":
    main()
