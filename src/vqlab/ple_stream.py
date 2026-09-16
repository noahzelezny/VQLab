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

THE FIX. The gather itself is tiny -- a few thousand rows of 160 -- so the
cost is entirely in holding the tables. After each shard is used, evaluate
the accumulator to cut the graph and REPLACE THE SHARD'S WEIGHT WITH A FRESH
LAZY HANDLE, which drops the materialised buffer. Peak falls from 96 GiB to
roughly one shard.

THIS CHANGES NO ARITHMETIC. Same ops, same order, same values; only the
evaluation boundary and the lifetime of the weights move. That matters
because this is the runtime our artifacts BUNDLE (TWO-RUNTIMES.md, F31), so
it is installed at RUNTIME on a loaded model, per process, and never edited
into the venv or into a shipped model.py.

THE COST is re-reading the tables once per call rather than once per load.
Put the teacher on fast storage before using this: on the archive HDD a
12k-token pass re-reads ~96 GiB per chunk.

    from ple_stream import install
    n = install(model, model_path)     # -> number of shard modules bounded
"""
import json
import os

import mlx.core as mx


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
    import numpy as np

    rows, dim = emb.rows, emb.dim

    def streamed(gid):
        flat = gid.reshape(-1)
        shard_of = np.array(flat // rows, copy=False)
        row_of = flat % rows
        out = mx.zeros((flat.size, dim), dtype=mx.float32)
        for s in np.unique(shard_of).tolist():
            sel = mx.array(np.nonzero(shard_of == s)[0])
            mod = getattr(emb, f"shard_{s}")
            e = mod(mx.take(row_of, sel))
            out = mx.put_along_axis(out, sel[:, None],
                                    e.astype(mx.float32), axis=0)
            mx.eval(out)                       # cut the graph here
            ent = keys.get(int(s))
            if ent is not None:                # drop the materialised table
                mod.weight = mx.load(ent[0])[ent[1]]
            mx.clear_cache()
        return out.reshape(*gid.shape, dim)

    emb.__call__ = streamed
    if verbose:
        print(f"[ple_stream] streaming gather installed on {path or '<root>'} "
              f"({emb.n_shards} shards)", flush=True)
