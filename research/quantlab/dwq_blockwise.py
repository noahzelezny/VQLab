#!/usr/bin/env python
"""Block-wise DWQ for models whose student doesn't fit in RAM (E18).

Why this exists: stock ``mlx_lm dwq`` holds the FULL student + an fp32 copy of
every trainable + Adam state (260 GiB for our 397B t2.1 vs the M4's 119 GiB;
the student ALONE is 122.9 GiB). ``--pipeline`` only splits the teacher.

This script distills ONE transformer block at a time (BRECQ-style sequential
with error correction):

  T_0 = teacher_embed(tokens)        S_0 = student_embed(tokens)
  for i in blocks:
      target  = teacher_block_i(T_i)          # pure-teacher trajectory
      train student_block_i scales/biases so student_block_i(S_i) -> target
      T_{i+1} = target
      S_{i+1} = trained_student_block_i(S_i)  # drift is corrected, then carried

Peak memory is ~one teacher block (bf16, ~12.5 GiB on the 397B) + one student
block + three activation streams + Adam for one block's scales/biases — ~25 GiB
total, so it runs with FULL Adam on either box. Used teacher/student blocks are
dereferenced (layers[i] = None) so memory stays flat across all 60 blocks.

E15 lessons honored here:
  * load(lazy=True) runs under mx.stream(mx.cpu) — lazy-load ops bind to the
    stream active at op CREATION; a GPU-ambient load stalls on SSD page-in and
    the Metal watchdog kills it.
  * Blocks are found by children-scan descent, never the delegating ``.layers``
    property (VLM wrappers forward it from the inner text model).
E16 lessons: trainables are affine scales/biases with bits < 8 (same unfreeze
predicate as dwq.py); output DROPS the vision sidecar + processor configs, so
``--assemble`` re-attaches them.

Two phases (resumable; patches are the checkpoint):

  train:    python dwq_blockwise.py train --student S --teacher T --patch-dir P
  assemble: python dwq_blockwise.py assemble --student S --patch-dir P --out O
"""

import argparse
import gc
import json
import math
import os
import shutil
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten


# ---------------------------------------------------------------- model bits

def find_core(model):
    """Children-scan descent to the container that OWNS the block list.

    Copied from optiq's streaming convert (E15 defect #5): trust only the
    children scan, never ``model.layers`` — wrappers delegate it as a property.
    """
    def holds_layers(m):
        if not isinstance(m, nn.Module):
            return False
        if isinstance(getattr(m, "layers", None), (list, tuple)):
            return True
        return any(
            holds_layers(c) for c in m.children().values()
            if not isinstance(c, (list, tuple))
        )

    core = model
    while not isinstance(core.children().get("layers"), (list, tuple)):
        descend = None
        for name, child in core.children().items():
            if isinstance(child, (list, tuple)) or name == "mtp":
                continue
            if holds_layers(child):
                descend = child
                break
        if descend is None:
            raise RuntimeError("no block list found by children scan")
        core = descend
    return core


def lazy_load(path):
    """mlx_lm load, lazy, on the CPU stream (E15 root-cause fix)."""
    from mlx_lm.utils import load
    with mx.stream(mx.cpu):
        model, tokenizer, config = load(path, return_config=True, lazy=True)
    return model, tokenizer, config


def unfreeze_trainables(block):
    """dwq.py's predicate verbatim: affine quantized layers with bits < 8."""
    def unfreeze(_, m):
        if (
            hasattr(m, "bits")
            and hasattr(m, "group_size")
            and getattr(m, "mode", None) == "affine"
            and m.bits < 8
        ):
            m.unfreeze(keys=["scales", "biases"], recurse=False)
    block.freeze()
    block.apply_to_modules(unfreeze)


# ---------------------------------------------------------------------- data

def build_batches(tokenizer, data_path, num_samples, seq_len, batch_size, seed):
    """Fixed-length packed batches: concatenate samples, chunk to seq_len.

    Packing (vs pad+mask) keeps every position valid so the hidden-state MSE
    needs no length mask, and end-padding subtleties in the ssm path never
    arise. Calibration only — sample boundaries mid-sequence are fine.
    """
    from mlx_lm.quant.dwq import load_data
    need_tokens = num_samples * seq_len
    # load_data returns (tokens, offset) pairs; over-fetch, then pack.
    train, _valid = load_data(
        tokenizer, data_path,
        num_samples=max(64, math.ceil(need_tokens / 120)),
        max_seq_length=seq_len + 1,
    )
    stream = []
    for tokens, _off in train:
        stream.extend(tokens)
        if len(stream) >= need_tokens:
            break
    if len(stream) < need_tokens:
        raise RuntimeError(
            f"corpus too small: {len(stream)} tokens < {need_tokens} needed"
        )
    arr = np.array(stream[:need_tokens], dtype=np.int32).reshape(num_samples, seq_len)
    return [
        mx.array(arr[i : i + batch_size])
        for i in range(0, num_samples, batch_size)
    ]


# ------------------------------------------------------------------ training

def forward_stream(block, acts, mask, desc):
    """Run every activation batch through a block in eval mode; return outputs."""
    block.eval()
    outs = []
    t0 = time.time()
    for a in acts:
        y = block(a, mask=mask, cache=None)
        mx.eval(y)
        outs.append(y)
    print(f"    {desc}: {len(acts)} batches in {time.time() - t0:.1f}s")
    return outs


def train_block(block, s_acts, targets, mask, *, lr, epochs, grad_ckpt=False,
                project=None, n_pos=64, seed=0):
    """Optimize one block's scales/biases (fp32 master weights + Adam).

    project=None  -> E18 objective: normalized MSE on the block's hidden state.
                     MEASURED HARMFUL at 397B scale (PPL 9.1 -> 37/47): hidden
                     MSE spends its budget on norm-heavy directions that the
                     logits don't care about, and bakes the calibration set's
                     drift into the weights.
    project=fn    -> E19 objective ("logit lens"): push BOTH the student and
                     teacher block outputs through the frozen final-norm +
                     lm_head and match their distributions with KL. Same
                     block-local memory, but the loss now only rewards changes
                     that move the actual next-token distribution.

    Only n_pos sequence positions per batch are projected — a full
    [B, S, vocab] logit tensor per step would dwarf the block itself, and a
    64-position sample is plenty of signal for scales/biases (positions are
    drawn once per batch index and held fixed so the objective is stationary).
    """
    unfreeze_trainables(block)
    block.train()
    trainable = block.trainable_parameters()
    n_train = sum(v.size for _, v in tree_flatten(trainable))
    if n_train == 0:
        print("    no trainables (block fully >=8-bit?) — skipped")
        return None, 0.0, 0.0

    # fp32 master copy (dwq.py pattern: tiny steps underflow in bf16)
    params = tree_map(lambda x: x.astype(mx.float32), trainable)
    native_dtype = tree_flatten(trainable)[0][1].dtype
    opt = optim.Adam(learning_rate=lr, bias_correction=True)

    pos_cache: dict[int, mx.array] = {}

    def _positions(bi, S):
        if bi not in pos_cache:
            g = np.random.default_rng(seed + bi)
            k = min(n_pos, S)
            pos_cache[bi] = mx.array(
                np.sort(g.choice(S, size=k, replace=False)).astype(np.int32))
        return pos_cache[bi]

    def loss_fn(p, x, target, bi=0):
        block.update(tree_map(lambda v: v.astype(native_dtype), p))
        y = block(x, mask=mask, cache=None)
        if project is None:
            d = (y - target).astype(mx.float32)
            t = target.astype(mx.float32)
            # normalized MSE: scale-free across depth, comparable block to block
            return (d * d).sum() / (t * t).sum()
        idx = _positions(bi, y.shape[1])
        s_log = nn.log_softmax(project(y[:, idx, :]).astype(mx.float32), axis=-1)
        t_log = mx.stop_gradient(
            nn.log_softmax(project(target[:, idx, :]).astype(mx.float32), axis=-1))
        # KL(teacher || student), averaged per projected position
        return (mx.exp(t_log) * (t_log - s_log)).sum() / (
            y.shape[0] * idx.shape[0])

    grad_fn = mx.value_and_grad(loss_fn)

    def epoch(train):
        tot = 0.0
        nonlocal params
        for bi, (x, tgt) in enumerate(zip(s_acts, targets)):
            if train:
                loss, grads = grad_fn(params, x, tgt, bi)
                params = opt.apply_gradients(grads, params)
                mx.eval(params, opt.state)
            else:
                loss = loss_fn(params, x, tgt, bi)
                mx.eval(loss)
            tot += loss.item()
        return tot / len(s_acts)

    init = epoch(train=False)
    t0 = time.time()
    final = init
    for e in range(epochs):
        final = epoch(train=True)
        print(f"    epoch {e + 1}/{epochs}: nmse {final:.5f} "
              f"(init {init:.5f}, {time.time() - t0:.0f}s)")
    if final > init:
        print("    ❌ loss got WORSE — reverting this block to its "
              "pre-DWQ scales/biases")
        return None, init, final
    block.update(tree_map(lambda v: v.astype(native_dtype), params))
    return tree_flatten(block.trainable_parameters()), init, final


# ------------------------------------------------------------------- phases

def cmd_train(args):
    patch_dir = Path(args.patch_dir)
    patch_dir.mkdir(parents=True, exist_ok=True)
    meta_path = patch_dir / "progress.json"
    progress = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    print(f"[1/3] loading student (lazy, cpu stream): {args.student}")
    student, tokenizer, s_cfg = lazy_load(args.student)
    print(f"[1/3] loading teacher (lazy, cpu stream): {args.teacher}")
    teacher, _, _ = lazy_load(args.teacher)

    s_core, t_core = find_core(student), find_core(teacher)
    n_blocks = len(s_core.layers)
    assert n_blocks == len(t_core.layers), "block count mismatch"
    last = min(n_blocks, args.last_block + 1 if args.last_block >= 0 else n_blocks)
    print(f"      {n_blocks} blocks; training [{args.first_block}, {last})")

    resume_from = 0
    if args.resume_acts:
        # Cross-machine split (E18b): another box ran [0, k], dumped the
        # embedding-relative activation streams at block k, and this box
        # picks up from k+1. The calibration batches themselves are never
        # needed here — the streams already encode them — so this MUST use
        # the identical --num-samples/--seq-len/--seed as the dump run or
        # the batch count/shape silently mismatches later saves.
        print(f"[2/3] loading activation checkpoint: {args.resume_acts}")
        ck = mx.load(args.resume_acts)
        n_batches = ck["s_acts"].shape[0]
        s_acts = [mx.array(ck["s_acts"][j]) for j in range(n_batches)]
        t_acts = [mx.array(ck["t_acts"][j]) for j in range(n_batches)]
        resume_from = int(ck["next_block"].item())
        mx.eval(s_acts, t_acts)
        print(f"      resuming at block {resume_from}, {n_batches} batches")
    else:
        print(f"[2/3] building calibration batches: {args.num_samples} x "
              f"{args.seq_len} from {args.data_path}")
        batches = build_batches(tokenizer, args.data_path, args.num_samples,
                                args.seq_len, args.batch_size, args.seed)

        # Embedding stage (block "-1"): both streams start from their own embed.
        with mx.stream(mx.cpu):
            mx.eval(s_core.embed_tokens.parameters(),
                    t_core.embed_tokens.parameters())
        s_acts = [s_core.embed_tokens(b) for b in batches]
        t_acts = [t_core.embed_tokens(b) for b in batches]
        mx.eval(s_acts, t_acts)

    project = None
    if args.loss == "logit":
        # Frozen logit lens: the student's OWN final norm + lm_head, never
        # trained, applied at every depth. Using the student's head (not the
        # teacher's) keeps the objective aligned with what this model will
        # actually compute at inference.
        head_norm = s_core.norm
        lm_head = getattr(student, "lm_head", None)
        embed = s_core.embed_tokens
        with mx.stream(mx.cpu):
            mx.eval(head_norm.parameters(),
                    lm_head.parameters() if lm_head is not None
                    else embed.parameters())
        def project(h):                                    # noqa: E306
            h = head_norm(h)
            return lm_head(h) if lm_head is not None else embed.as_linear(h)
        print("      objective: LOGIT-LENS KL (E19) via frozen norm+lm_head")
    else:
        print("      objective: hidden-state MSE (E18 — known harmful at 397B)")

    print("[3/3] block loop")
    for i in range(n_blocks):
        if i < resume_from:
            continue
        if i >= last:
            break
        blk_s, blk_t = s_core.layers[i], t_core.layers[i]
        mask = None if blk_s.is_linear else "causal"
        patch_file = patch_dir / f"block_{i:03d}.safetensors"
        t0 = time.time()
        print(f"  block {i}/{n_blocks - 1} "
              f"({'linear' if blk_s.is_linear else 'full'}-attn)")

        # Materialize this block only (cpu stream — E15).
        with mx.stream(mx.cpu):
            mx.eval(blk_t.parameters(), blk_s.parameters())

        # Teacher trajectory: always needed to advance T (and is the target).
        t_next = forward_stream(blk_t, t_acts, mask, "teacher fwd")

        skip = i < args.first_block
        if not skip and patch_file.exists():
            print("    patch exists — applying (resume)")
            blk_s.update(tree_unflatten(list(mx.load(str(patch_file)).items())))
            skip = True
        if not skip:
            patch, init, final = train_block(
                blk_s, s_acts, t_next, mask,
                lr=args.learning_rate, epochs=args.epochs,
                project=project, n_pos=args.n_pos, seed=args.seed)
            if patch is not None:
                mx.save_safetensors(str(patch_file), dict(patch))
            progress[str(i)] = {
                "nmse_init": init, "nmse_final": final,
                "trained": patch is not None,
                "seconds": round(time.time() - t0, 1),
            }
            meta_path.write_text(json.dumps(progress, indent=1))

        # Student trajectory advances through the (possibly trained) block.
        s_next = forward_stream(blk_s, s_acts, mask, "student fwd")

        # Flat memory: dead blocks and dead activation streams get dropped.
        s_acts, t_acts = s_next, t_next
        s_core.layers[i] = None
        t_core.layers[i] = None
        del blk_s, blk_t
        gc.collect()
        mx.clear_cache()
        print(f"    done in {time.time() - t0:.0f}s "
              f"(peak {mx.get_peak_memory() / 1024**3:.1f}G)")

        if args.dump_acts_after is not None and i == args.dump_acts_after:
            ck_path = args.dump_acts_path or str(
                patch_dir / f"acts_after_{i:03d}.safetensors")
            mx.save_safetensors(ck_path, {
                "s_acts": mx.stack(s_acts),
                "t_acts": mx.stack(t_acts),
                "next_block": mx.array(i + 1),
            })
            print(f"    dumped activation checkpoint -> {ck_path} "
                  f"(hand this + --resume-acts to the other box)")
            break

    print(f"\ntrain phase complete — patches in {patch_dir}")
    print("next: python dwq_blockwise.py assemble --student ... "
          f"--patch-dir {patch_dir} --out ...")


def cmd_assemble(args):
    """Rewrite student shards with patched scales/biases; re-attach vision."""
    src, out = Path(args.student), Path(args.out)
    # --patch-dir accepts a comma list so a cross-machine split (E18b) can be
    # assembled from both boxes' patch dirs in one pass — block ranges are
    # disjoint by construction (--first-block/--last-block on each half), so
    # last-writer-wins on a collision would silently hide a re-run mistake;
    # assert disjointness instead.
    patch_dirs = [Path(p) for p in args.patch_dir.split(",")]
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing {out}")
    out.mkdir(parents=True)

    # Patched tensors, fully-qualified. Patches are saved from the block's
    # trainable_parameters() so keys are relative to the block; qualify with
    # the student's real prefix taken from its weight index.
    index = json.loads((src / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    some_layer_key = next(k for k in weight_map if ".layers.0." in k)
    prefix = some_layer_key.split(".layers.")[0]

    patched = {}
    seen_blocks = set()
    n_blocks_total = 0
    for patch_dir in patch_dirs:
        pfs = sorted(patch_dir.glob("block_*.safetensors"))
        for pf in pfs:
            i = int(pf.stem.split("_")[1])
            if i in seen_blocks:
                raise SystemExit(
                    f"block {i} patched in more than one --patch-dir "
                    f"({patch_dir} collides with an earlier dir) — merging "
                    f"would silently pick one at random")
            seen_blocks.add(i)
            for k, v in mx.load(str(pf)).items():
                patched[f"{prefix}.layers.{i}.{k}"] = v
        n_blocks_total += len(pfs)
    missing = [k for k in patched if k not in weight_map]
    if missing:
        raise SystemExit(f"patch keys not in student index, e.g. {missing[:3]}")
    print(f"{len(patched)} patched tensors across {n_blocks_total} blocks "
          f"from {len(patch_dirs)} patch dir(s)")

    shards = sorted(set(weight_map.values()))
    n_hit = 0
    for shard in shards:
        (out / shard).parent.mkdir(parents=True, exist_ok=True)
        with mx.stream(mx.cpu):
            tensors = mx.load(str(src / shard))
            hits = [k for k in tensors if k in patched]
            for k in hits:
                assert tensors[k].shape == patched[k].shape
                assert tensors[k].dtype == patched[k].dtype
                tensors[k] = patched[k]
            n_hit += len(hits)
            mx.save_safetensors(str(out / shard), tensors,
                                metadata={"format": "mlx"})
        del tensors
        gc.collect()
        print(f"  {shard}: {len(hits)} tensors patched")
    assert n_hit == len(patched), f"applied {n_hit} != {len(patched)} patched"

    # Same shapes/dtypes -> index is unchanged; copy configs + tokenizer +
    # optiq dir + THE VISION SIDECAR + processor configs (E16: dwq drops them).
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    for f in src.iterdir():
        if f.name.endswith(".safetensors") or f.name == "model.safetensors.index.json":
            continue
        if f.is_dir():
            # dirs (e.g. optiq/) may already hold rewritten shards — keep those
            shutil.copytree(
                f, out / f.name, dirs_exist_ok=True,
                copy_function=lambda s, d: (None if os.path.exists(d)
                                            else shutil.copy2(s, d)),
            )
        else:
            shutil.copy2(f, out / f.name)
    for req in ("optiq/optiq_vision.safetensors", "preprocessor_config.json",
                "video_preprocessor_config.json"):
        if not (out / req).exists():
            print(f"  ⚠️  {req} missing from output — attach before serving")
    print(f"\nassembled: {out}")


def cmd_merge(args):
    """Average per-block patches from N independent same-init runs (E18b).

    Both runs start from the same student and take small Adam steps around
    the same init (measured drift ~1%), so element-wise averaging lands in
    the flat basin between two very-nearby solutions — the model-soup /
    federated-averaging regime. --weights biases the average when runs used
    different sample counts (e.g. "512,256" -> 2:1).
    """
    dirs = [Path(p) for p in args.patch_dirs.split(",")]
    weights = ([float(w) for w in args.weights.split(",")]
               if args.weights else [1.0] * len(dirs))
    assert len(weights) == len(dirs), "--weights count must match dirs"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    names = [sorted(p.name for p in d.glob("block_*.safetensors"))
             for d in dirs]
    common = set(names[0]).intersection(*names[1:])
    only = {d: set(n) - common for d, n in zip(dirs, names)}
    for d, extra in only.items():
        if extra:
            print(f"  ⚠️  {d} has blocks the others lack (NOT merged, "
                  f"copy by hand if intended): {sorted(extra)}")

    total = sum(weights)
    for name in sorted(common):
        tensor_sets = [mx.load(str(d / name)) for d in dirs]
        keys = set(tensor_sets[0])
        assert all(set(t) == keys for t in tensor_sets), f"key mismatch {name}"
        merged = {}
        for k in keys:
            acc = sum(t[k].astype(mx.float32) * w
                      for t, w in zip(tensor_sets, weights)) / total
            merged[k] = acc.astype(tensor_sets[0][k].dtype)
        mx.save_safetensors(str(out / name), merged)
    print(f"merged {len(common)} blocks from {len(dirs)} runs "
          f"(weights {weights}) -> {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--student", required=True)
    t.add_argument("--teacher", required=True)
    t.add_argument("--patch-dir", required=True)
    t.add_argument("--data-path", default="allenai/tulu-3-sft-mixture")
    t.add_argument("--num-samples", type=int, default=256)
    t.add_argument("--seq-len", type=int, default=512)
    t.add_argument("--batch-size", type=int, default=4)
    t.add_argument("--epochs", type=int, default=2)
    t.add_argument("--learning-rate", type=float, default=1e-5)
    t.add_argument("--seed", type=int, default=123)
    t.add_argument("--first-block", type=int, default=0,
                   help="first block to TRAIN (earlier blocks fwd-only)")
    t.add_argument("--last-block", type=int, default=-1,
                   help="last block to train, inclusive; -1 = all. The loop "
                        "STOPS after this block (dry runs end early).")
    t.add_argument("--loss", choices=["mse", "logit"], default="mse",
                   help="mse = E18 hidden-state MSE (measured HARMFUL at 397B). "
                        "logit = E19 logit-lens KL through the frozen "
                        "final-norm + lm_head.")
    t.add_argument("--n-pos", type=int, default=64,
                   help="sequence positions per batch projected to logits "
                        "(--loss logit only)")
    t.add_argument("--dump-acts-after", type=int, default=None,
                   help="cross-machine split (E18b): after finishing this "
                        "block, save the activation streams + exit instead "
                        "of continuing. Hand the file to the other box's "
                        "--resume-acts.")
    t.add_argument("--dump-acts-path", type=str, default=None,
                   help="override the checkpoint path (default: "
                        "<patch-dir>/acts_after_NNN.safetensors)")
    t.add_argument("--resume-acts", type=str, default=None,
                   help="path from --dump-acts-after on ANOTHER box. Loads "
                        "the activation streams and starts at next_block "
                        "instead of embedding+block-0. MUST use the same "
                        "--num-samples/--seq-len/--seed as the dump run.")
    t.set_defaults(fn=cmd_train)

    m = sub.add_parser("merge")
    m.add_argument("--patch-dirs", required=True,
                   help="comma-separated patch dirs from independent runs")
    m.add_argument("--weights", default=None,
                   help="comma-separated weights, e.g. '512,256' (default 1:1)")
    m.add_argument("--out", required=True)
    m.set_defaults(fn=cmd_merge)

    a = sub.add_parser("assemble")
    a.add_argument("--student", required=True)
    a.add_argument("--patch-dir", required=True,
                   help="comma-separated for a cross-machine split (E18b), "
                        "e.g. m3-patches,m4-patches")
    a.add_argument("--out", required=True)
    a.set_defaults(fn=cmd_assemble)

    args = p.parse_args()
    np.random.seed(getattr(args, "seed", 123))
    mx.random.seed(getattr(args, "seed", 123))
    args.fn(args)


if __name__ == "__main__":
    main()
