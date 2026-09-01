"""Extract a model's MTP head from its source checkpoint into one graft file.

    python -m vqlab.cli mtp-extract --src <bf16-checkpoint> --out graft.safetensors

MLX conversion strips `mtp.*` systematically -- Qwen's own MLX uploads carry
none, and neither do ours -- so the head only exists in the original
checkpoint. This pulls it out into the single-file graft that `mtp-pack` and
`mtp-graft` consume, which is otherwise a step that happened once, by hand,
and was not reproducible.

Only the shards that actually hold `mtp.*` are opened, and only those tensors
are materialized, so extracting a 12 GiB head from a 400B checkpoint costs
12 GiB of RAM and not 800.

Per CONTRIBUTING, the load -> save path takes the lazy-read cure: the read
happens inside `mx.stream(mx.cpu)` with an `mx.eval` in the same block, and
the result is asserted non-zero before writing. A lazily-read tensor that is
evaluated after the stream closes silently writes zeros, and a graft of zeros
produces a head with exactly 0.0 acceptance -- indistinguishable from the
RMSNorm bug this package already documents.
"""
import argparse
import json
import pathlib
import re
import sys
from collections import defaultdict

import mlx.core as mx

MTP_KEY = re.compile(r"(^|\.)(mtp|nextn)\b", re.IGNORECASE)


def find_mtp_keys(src: pathlib.Path):
    """{shard: [keys]} for every MTP tensor, from the index."""
    idx = src / "model.safetensors.index.json"
    if not idx.exists():
        raise SystemExit(f"no {idx.name} in {src}")
    wm = json.loads(idx.read_text())["weight_map"]
    by_shard = defaultdict(list)
    for k, shard in wm.items():
        if MTP_KEY.search(k):
            by_shard[shard].append(k)
    return dict(by_shard), len(wm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source checkpoint directory")
    ap.add_argument("--out", required=True, help="graft safetensors to write")
    ap.add_argument("--strip-prefix", default=None,
                    help="drop this leading prefix from every key "
                         "(default: keep keys as found)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be extracted and stop")
    a = ap.parse_args()

    src = pathlib.Path(a.src)
    by_shard, n_total = find_mtp_keys(src)
    n_mtp = sum(len(v) for v in by_shard.values())
    if not n_mtp:
        raise SystemExit(
            f"no mtp/nextn tensors in {src}. The MLX conversion almost "
            f"certainly stripped them -- extract from the original "
            f"(usually bf16) checkpoint instead.")
    print(f"{src.name}: {n_mtp} MTP tensors of {n_total}, in "
          f"{len(by_shard)} shard(s)")
    groups = defaultdict(int)
    for keys in by_shard.values():
        for k in keys:
            groups[".".join(k.split(".")[:4])] += 1
    for g, n in sorted(groups.items(), key=lambda x: -x[1])[:8]:
        print(f"    {g:<52} x{n}")
    if a.dry_run:
        return 0

    out = {}
    total = 0
    for shard in sorted(by_shard):
        keys = by_shard[shard]
        # Lazy-read cure: read AND evaluate inside the cpu stream, or the
        # deferred read can resolve after the stream closes and write zeros.
        with mx.stream(mx.cpu):
            part = mx.load(str(src / shard))
            picked = {k: part[k] for k in keys}
            mx.eval(list(picked.values()))
        for k, v in picked.items():
            name = k
            if a.strip_prefix and name.startswith(a.strip_prefix):
                name = name[len(a.strip_prefix):]
            out[name] = v
            total += v.nbytes
        del part, picked
        print(f"  {shard}: +{len(keys)} tensors "
              f"({total / 2**30:.2f} GiB so far)", flush=True)

    # A graft of zeros yields a head with exactly 0.0 acceptance, which is
    # indistinguishable from the RMSNorm-convention bug. Refuse to write one.
    dead = [k for k, v in out.items()
            if v.size and not bool(mx.any(v != 0).item())]
    if dead:
        raise SystemExit(
            f"FAIL: {len(dead)} extracted tensors are entirely zero, e.g. "
            f"{dead[:3]}. This is the lazy-read failure, not a real head.")

    outp = pathlib.Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(outp), out, metadata={
        "format": "mlx", "vqlab_mtp_graft": json.dumps(
            {"source": src.name, "tensors": len(out)})})
    print(f"\nwrote {outp}")
    print(f"  {len(out)} tensors, {outp.stat().st_size / 2**30:.2f} GiB")
    print(f"  key prefixes: {sorted({k.split('.')[0] for k in out})}")
    print(f"  all-zero check: PASS ({len(out)} tensors carry data)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
