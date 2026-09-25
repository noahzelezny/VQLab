#!/usr/bin/env python
"""active-bytes — weight bytes READ PER DECODE TOKEN, by component.

The roofline denominator. A decode step reads every dense weight once and
only the routed experts' slices; this walks the safetensors headers (no
tensor is loaded, no GPU is touched) and splits the total into components,
so "how many bytes does one token cost" stops being a guess.

Motivated by F22, whose bytes/token accounting for Flash covered the expert
matrices and therefore understated a hyper-connection trunk that is not
expert-shaped. Effective-bandwidth rows computed against an understated
denominator read as a "large fixed cost" that may simply be unaccounted
bytes -- so the denominator gets its own instrument.

Reads config.json + the safetensors header(s). Metadata only: safe to run
on a contended box, and it never loads a tensor.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

# dtype -> bytes/element, as safetensors spells them
_ITEMSIZE = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
    "U32": 4, "I32": 4, "F32": 4,
    "U64": 8, "I64": 8, "F64": 8,
}


def _read_header(path: Path) -> dict:
    """The safetensors header JSON: 8-byte LE length, then that many bytes."""
    with path.open("rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        return json.loads(fh.read(n))


def _tensors(art: Path):
    """Yield (name, nbytes) for every tensor, from headers alone."""
    index = art / "model.safetensors.index.json"
    files = sorted({v for v in json.loads(index.read_text())["weight_map"].values()}) \
        if index.exists() else ["model.safetensors"]
    for fname in files:
        fpath = art / fname
        if not fpath.exists():
            raise SystemExit(f"missing shard: {fpath}")
        for name, meta in _read_header(fpath).items():
            if name == "__metadata__":
                continue
            size = _ITEMSIZE.get(meta["dtype"])
            if size is None:
                raise SystemExit(f"unknown dtype {meta['dtype']} on {name}")
            shape = meta["shape"]
            n = 1
            for d in shape:
                n *= d
            yield name, n * size, shape


# How a tensor is READ on a decode step, which is not the same as how big
# it is. Three modes:
#   dense    - every byte, every token (a matmul weight)
#   routed   - top_k of num_experts slices (the MoE expert stacks)
#   gathered - a handful of ROWS (embedding and ngram lookup tables). Billing
#              these dense is the error that made the first cut of this tool
#              report 9.6 GB/token of PLE: a table you index is not a table
#              you read.
DENSE, ROUTED, GATHERED = "dense", "routed", "gathered"


def _component(name: str) -> tuple[str, str]:
    """Bucket a tensor into (component, read-mode).

    Ordered: the first match wins, so the specific names (the gathered
    lookup tables, the always-on shared expert) are tested before the
    generic catch-alls they would otherwise fall into.
    """
    n = name
    # qwen4_exp spells the tower `model.visual.*`, so a startswith() test
    # silently billed 0.9 GB/token of idle vision weights as dense trunk.
    if n.startswith("visual") or ".visual." in n or "vision" in n:
        return "vision tower (idle on text)", GATHERED  # untouched: 0 rows
    # --- gathered lookup tables: indexed per token, never read whole ---
    if "ple_embedding" in n:
        return "PLE ngram banks (gathered)", GATHERED
    if "embed_tokens" in n:
        return "embedding (gathered)", GATHERED
    # --- routed: only the chosen experts' slices ---
    # `shared_expert` must be tested FIRST: it lives under mlp.* but is
    # DENSE -- every token runs it, unrouted.
    if "shared_expert_gate" in n:
        return "router + shared gate", DENSE
    if "shared_expert" in n:
        return "shared expert (dense MLP)", DENSE
    if "switch_mlp" in n or "experts" in n:
        return "experts (MoE, routed)", ROUTED
    if n.endswith("mlp.gate.weight"):
        return "router + shared gate", DENSE
    # --- dense trunk ---
    if "hyper_connection" in n or "mix_weight" in n or "inject_weight" in n:
        return "hyper-connections", DENSE
    if ".ple." in n or "ngram" in n:
        return "PLE projections", DENSE
    if "linear_attn" in n:
        return "GatedDeltaNet (linear attn)", DENSE
    if "self_attn" in n or "indexer" in n:
        return "full attention", DENSE
    if "lm_head" in n:
        return "lm_head", DENSE
    if "norm" in n:
        return "norms", DENSE
    # Anything unrecognised is billed DENSE on purpose: an unknown tensor
    # should show up as a fat "other" row demanding a name, not vanish.
    return "other", DENSE


def main() -> int:
    ap = argparse.ArgumentParser(
        description="weight bytes read per decode token, by component")
    ap.add_argument("artifact", help="artifact directory")
    ap.add_argument("--peak-gbs", type=float, default=819.0,
                    help="peak memory bandwidth GB/s for the roofline row")
    ap.add_argument("--measured-ms", type=float, default=None,
                    help="measured ms/token, to print the achieved fraction")
    ap.add_argument("--gather-rows", type=int, default=8,
                    help="rows read per token from each gathered lookup "
                         "table (embedding=1; PLE reads one row per ngram "
                         "head). Deliberately generous: the point is that "
                         "gathered tables are negligible, and an overestimate "
                         "that stays negligible proves it.")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    art = Path(a.artifact)
    cfg = json.loads((art / "config.json").read_text())
    text = cfg.get("text_config", cfg)
    n_exp = text.get("num_experts") or text.get("num_local_experts")
    top_k = text.get("num_experts_per_tok")
    if not n_exp or not top_k:
        print("note: no MoE routing in config; treating every tensor as dense",
              file=sys.stderr)
        n_exp = top_k = 1
    routed = top_k / n_exp

    resident, active = {}, {}
    for name, nbytes, shape in _tensors(art):
        comp, mode = _component(name)
        resident[comp] = resident.get(comp, 0) + nbytes
        if mode == ROUTED:
            billed = nbytes * routed
        elif mode == GATHERED:
            if comp.startswith("vision"):
                billed = 0.0          # text-only decode never touches it
            else:
                rows = 1 if "embedding (gathered)" == comp else a.gather_rows
                nrows = shape[0] if shape else 1
                billed = nbytes * min(rows, nrows) / max(nrows, 1)
        else:
            billed = nbytes
        active[comp] = active.get(comp, 0) + billed

    total_active = sum(active.values())
    total_resident = sum(resident.values())
    floor_ms = total_active / (a.peak_gbs * 1e9) * 1e3

    if a.json:
        print(json.dumps({
            "artifact": str(art), "num_experts": n_exp,
            "experts_per_tok": top_k,
            "resident_bytes": resident, "active_bytes_per_token": active,
            "total_active_bytes_per_token": total_active,
            "total_resident_bytes": total_resident,
            "bandwidth_floor_ms_per_token": floor_ms,
            "peak_gbs": a.peak_gbs,
        }, indent=2))
        return 0

    print(f"\n{art.name}")
    print(f"  MoE routing: top-{top_k} of {n_exp} experts "
          f"({routed:.2%} of expert bytes per token)\n")
    print(f"  {'component':<30} {'resident':>10} {'per token':>11} {'share':>7}")
    print(f"  {'-'*30} {'-'*10} {'-'*11} {'-'*7}")
    for comp in sorted(active, key=lambda c: -active[c]):
        share = active[comp] / total_active * 100 if total_active else 0.0
        print(f"  {comp:<30} {resident[comp]/1e9:>9.2f}G "
              f"{active[comp]/1e9:>10.3f}G {share:>6.1f}%")
    print(f"  {'-'*30} {'-'*10} {'-'*11} {'-'*7}")
    print(f"  {'TOTAL':<30} {total_resident/1e9:>9.2f}G "
          f"{total_active/1e9:>10.3f}G {100.0:>6.1f}%")

    print(f"\n  bandwidth floor at {a.peak_gbs:.0f} GB/s peak: "
          f"{floor_ms:.3f} ms/token  ({1e3/floor_ms:,.0f} tok/s)")
    if a.measured_ms:
        print(f"  measured:                              "
              f"{a.measured_ms:.3f} ms/token  "
              f"({1e3/a.measured_ms:,.1f} tok/s)")
        print(f"  achieved fraction of peak bandwidth:   "
              f"{floor_ms/a.measured_ms*100:.1f}%")
        print(f"  headroom to the bandwidth roofline:    "
              f"{a.measured_ms/floor_ms:.1f}x")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
