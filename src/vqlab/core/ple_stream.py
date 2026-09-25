"""Make qwen4_exp's PLE n-gram tables STREAM instead of going resident.

THE PROBLEM. `_ShardedEmbedding.__call__` decides host-side which shards a
batch of n-gram ids touches, then gathers from each one:

    for s in touched:
        emb = getattr(self, f"shard_{s}")(mx.take(row_of, sel))
        out = mx.put_along_axis(out, sel[:, None], emb, axis=0)

Every touched shard is fully materialised for its lookup and stays
materialised, because the module holds the reference; and the whole chain is
one unevaluated graph until the caller evals. On Flash-Next-bf16 that is 128
shards of [2500012, 160] = 96 GiB, and at any realistic probe length ALL 128
are touched. The result is either MLX's memory-limit spin or, once the eval
is deferred, a single command buffer big enough for the Metal watchdog to
kill (F115 measured both).

THE FIX. The gather is TINY -- a few thousand rows of 160 values, about a
megabyte -- so nothing about the lookup needs 96 GiB; only holding the tables
does. Rather than materialise a shard to read a few rows out of it, mmap the
shard's bytes and index the rows directly, so the OS pages in the ~4 KB per
row actually touched and nothing else. bf16 is widened to float32 with a
uint16 -> uint32 << 16 view, which is exact (bf16 is the top half of a
float32).

A FIRST VERSION OF THIS FILE materialised each shard, used it, then replaced
its weight with a fresh lazy handle to free the buffer. That bounded memory
correctly and was catastrophic for time: the tables are re-read ON EVERY
CALL, and the caller chunks the sequence, so a 12288-token pass at chunk 512
re-read 96 GiB twenty-four times -- 2.3 TB of I/O to use 24 MB of rows.
Bounding memory is not the same as not wasting work.

THIS CHANGES NO ARITHMETIC. The same rows are gathered into the same
positions; only where the bytes come from changes. That matters because this
is the runtime our artifacts BUNDLE (TWO-RUNTIMES.md, F31), so it is
installed at RUNTIME on a loaded model, per process, and never edited into
the venv or into a shipped model.py.

    from ple_stream import install
    n = install(model, model_path)     # -> number of shard modules bounded
"""
import json
import os

import time

import mlx.core as mx

STATS = {"calls": 0, "rows": 0, "secs": 0.0}


def _index(model_path):
    p = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))["weight_map"]


def _find_sharded(model):
    """Yield (owner_module, attr_name_prefix) for every _ShardedEmbedding."""
    seen, out = set(), []

    def walk(mod, path):
        if id(mod) in seen:
            return
        seen.add(id(mod))
        if hasattr(mod, "n_shards") and hasattr(mod, "rows") \
                and hasattr(mod, "shard_0"):
            out.append((mod, path))
            return
        children = getattr(mod, "children", None)
        items = children().items() if callable(children) else []
        for name, ch in items:
            if isinstance(ch, (list, tuple)):
                for i, c in enumerate(ch):
                    if hasattr(c, "children"):
                        walk(c, f"{path}.{name}.{i}" if path else f"{name}.{i}")
            elif hasattr(ch, "children"):
                walk(ch, f"{path}.{name}" if path else name)

    walk(model, "")
    return out


def install(model, model_path, verbose=True):
    """Install the streaming gather on every sharded n-gram table. Idempotent."""
    wmap = _index(model_path)
    if wmap is None:
        raise SystemExit(f"FAIL: no safetensors index under {model_path}; "
                         "cannot rebuild lazy handles for the PLE shards.")
    targets = _find_sharded(model)
    if not targets:
        return 0

    bound = 0
    for emb, path in targets:
        keys = {}
        for k, f in wmap.items():
            if ".ngram_embedding.shard_" not in k or not k.endswith(".weight"):
                continue
            s = k.split(".ngram_embedding.shard_")[1].split(".")[0]
            if s.isdigit():
                keys[int(s)] = (os.path.join(model_path, f), k)
        if not keys:
            # A QUANTIZED rung stores its shards as .codes/.codebook, not
            # .weight, so there is no lazy handle of that name to rebuild --
            # and it does not need one: a quantized PLE is ~13 GiB, not 96.
            # SAY SO. Returning a truthy "found" count while binding nothing
            # is how an instrument silently does not run (F115's alloc-sweep
            # None, same shape).
            if verbose:
                print(f"[ple_stream] {path or '<root>'}: found "
                      f"{emb.n_shards} shards but no '.weight' tensors in the "
                      f"index (quantized rung?) -- NOT streamed, left as is",
                      flush=True)
            continue
        _bind(emb, keys, verbose, path)
        bound += 1
    if verbose and not bound:
        print(f"[ple_stream] {len(targets)} sharded table(s) found, "
              f"0 streamed", flush=True)
    return bound


def _bind(emb, keys, verbose, path):
    """Install the row-wise gather ON THE CLASS, not the instance.

    `emb.__call__ = f` DOES NOT WORK and fails silently: Python resolves
    special methods on the TYPE, so `emb(gid)` still reaches the original
    `type(emb).__call__`. The first version of this file did exactly that,
    reported success, and never ran -- the teacher pass it was credited with
    unblocking had actually been unblocked by the eager-eval budget alone.
    Measured the only way it could be: a call counter that read zero.

    The class is patched once and dispatches per instance, so an instance
    without a key map keeps the original behaviour.
    """
    import numpy as np

    cls = type(emb)
    emb._ps_keys = keys
    emb._ps_readers = {}

    if getattr(cls, "_ps_patched", False):
        if verbose:
            print(f"[ple_stream] row-wise gather enabled on "
                  f"{path or '<root>'} ({emb.n_shards} shards)", flush=True)
        return

    orig = cls.__call__

    def reader(self, s_idx):
        if s_idx not in self._ps_readers:
            fpath, key = self._ps_keys[s_idx]
            with open(fpath, "rb") as fh:
                hlen = int.from_bytes(fh.read(8), "little")
                hdr = json.loads(fh.read(hlen))
            info = hdr[key]
            if info["dtype"] not in ("BF16", "F16"):
                raise SystemExit(f"FAIL: {key} is {info['dtype']}; "
                                 f"ple_stream handles BF16/F16 only.")
            start = 8 + hlen + info["data_offsets"][0]
            mm = np.memmap(fpath, dtype=np.uint16, mode="r", offset=start,
                           shape=tuple(info["shape"]))
            self._ps_readers[s_idx] = (mm, info["dtype"])
        return self._ps_readers[s_idx]

    def patched(self, gid):
        if not hasattr(self, "_ps_keys"):
            return orig(self, gid)
        t0 = time.time()
        rows, dim = self.rows, self.dim
        flat = gid.reshape(-1)
        shard_of = np.array(flat // rows, copy=False)
        row_of = np.array(flat % rows, copy=False)
        out = np.zeros((flat.size, dim), dtype=np.float32)
        for s_idx in np.unique(shard_of).tolist():
            sel = np.nonzero(shard_of == s_idx)[0]
            s_idx = int(s_idx)
            if s_idx not in self._ps_keys:
                raise SystemExit(f"FAIL: shard {s_idx} missing from the "
                                 f"index; refusing to skip rows silently.")
            mm, dt = reader(self, s_idx)
            raw = np.asarray(mm[row_of[sel]])   # pages in only these rows
            if dt == "BF16":                    # exact: bf16 is the high half
                out[sel] = (raw.astype(np.uint32) << 16).view(np.float32)
            else:
                out[sel] = raw.view(np.float16).astype(np.float32)
        STATS["calls"] += 1
        STATS["rows"] += int(flat.size)
        STATS["secs"] += time.time() - t0
        return mx.array(out).reshape(*gid.shape, dim)

    cls.__call__ = patched
    cls._ps_patched = True
    if verbose:
        print(f"[ple_stream] row-wise gather installed on {cls.__name__} "
              f"for {path or '<root>'} ({emb.n_shards} shards)", flush=True)
