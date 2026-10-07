#!/usr/bin/env python
"""stage-bandwidth — achieved memory bandwidth of EVERY decode stage.

The two halves already existed and nothing joined them. `decode-timeline`
partitions one decode token's time into stages (Lnn.attn / Lnn.gdn /
Lnn.mlp / embed / final_norm / lm_head); `active-bytes` bills the weight
bytes a decode step reads, but by COMPONENT, summed over the whole network.
Bytes / time per stage is the number that says WHERE a token's time is
spent above the bandwidth roofline -- and the lab's open decode question
(F135/F190: VQ moves weights at ~346-453 GB/s where affine 8-bit reaches
~626 on the same box, so the loss is overhead, not physics) is exactly
"which stages run below the roofline, and by how many ms".

For each stage:

    bytes      weight bytes the stage reads per token, billed the way
               active-bytes bills them (dense / routed top-k / gathered rows)
    ms         the stage's time, from a decode-timeline --json-out file
    GB/s       bytes / ms
    floor ms   bytes / peak bandwidth: the fastest this stage can be
    excess ms  ms - floor: the time the stage spends ABOVE the roofline.
               This is the ranking. A stage with few bytes and many ms is
               pure overhead (launches, sync, small-op chains); a stage near
               its floor has nothing left to give.

Headers and a JSON file only: no tensor is loaded, no GPU is touched, so it
runs anywhere, any time, after the GPU run that produced the timeline.

WHAT IT DOES NOT BILL. The KV cache an attention stage reads grows with the
context; it is not a weight and is not in the bytes column. At the
timeline's default context (64) it is negligible against the weights; at
long context the attention rows' GB/s understate their real traffic. The
context the timeline ran at is printed beside the table for that reason.

    vqlab decode-timeline --art <dir> --json-out t.json      (on the GPU box)
    vqlab stage-bandwidth <dir> --timeline t.json --peak-gbs 546
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from vqlab.bench.active_bytes import GATHERED, ROUTED, _component, _routing, _tensors
from vqlab.core.artifact import tensor_class

_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")
# the MLP half of a layer: dense MLP, MoE router, routed and shared experts,
# and the norm that feeds them (decode-timeline's Lnn.mlp stage runs
# post_attention_layernorm + mlp)
_MLP_HALF = ("mlp.", "experts", "switch_mlp", "shared_expert", "moe.",
             "block_sparse_moe", "post_attention_layernorm")
_STAGE = re.compile(r"^L(\d+)\.(\w+)$")


def stage_of(name: str) -> str:
    """The decode-timeline stage that reads this tensor, spelled the way
    decode-timeline spells it, with the attention half as `Lnn.attn` (the
    timeline's gdn/attn distinction is a property of the layer, matched
    later). 'unused' for what a text decode step never reads."""
    cls = tensor_class(name)
    if cls in ("tower", "mtp"):
        return "unused"
    m = _LAYER.search(name)
    if m:
        i = int(m.group(1))
        half = "mlp" if any(k in name for k in _MLP_HALF) else "attn"
        return f"L{i:02d}.{half}"
    if "embed_tokens" in name:
        return "embed"
    if "lm_head" in name:
        return "lm_head"
    if "norm" in name:
        return "final_norm"
    return "other"


def bytes_by_stage(art: Path, gather_rows: int = 8) -> dict[str, float]:
    """Bytes per decode token, per stage, billed as active-bytes bills them."""
    cfg = json.loads((art / "config.json").read_text())
    n_exp, top_k = _routing(cfg)
    routed = (top_k / n_exp) if n_exp and top_k else 1.0
    out: dict[str, float] = {}
    for name, nbytes, shape in _tensors(art):
        stage = stage_of(name)
        _comp, mode = _component(name)
        if mode == ROUTED:
            billed = nbytes * routed
        elif mode == GATHERED:
            rows = 1 if "embed_tokens" in name else gather_rows
            nrows = shape[0] if shape else 1
            billed = 0.0 if stage == "unused" else \
                nbytes * min(rows, nrows) / max(nrows, 1)
        else:
            billed = nbytes
        out[stage] = out.get(stage, 0.0) + billed
    return out


def _key(stage_name: str) -> str:
    """A timeline stage name -> the key bytes_by_stage uses (gdn -> attn)."""
    m = _STAGE.match(stage_name)
    if m and m.group(2) != "mlp":
        return f"L{int(m.group(1)):02d}.attn"
    return stage_name


def join(timeline: dict, byts: dict[str, float],
         peak_gbs: float | None) -> dict:
    """One row per timed stage, plus the bytes no timed stage claimed."""
    rows, claimed = [], set()
    for st in timeline["stages"]:
        k = _key(st["name"])
        b = byts.get(k, 0.0)
        claimed.add(k)
        ms = float(st["ms"])
        row = {"name": st["name"], "kind": st.get("kind") or st["name"],
               "bytes": b, "ms": ms,
               "gbs": (b / (ms * 1e6)) if ms > 0 else None}
        if peak_gbs:
            floor = b / (peak_gbs * 1e6)
            row["floor_ms"] = floor
            row["excess_ms"] = ms - floor
        rows.append(row)
    unclaimed = {k: v for k, v in byts.items()
                 if k not in claimed and k != "unused" and v > 0}
    return {"rows": rows, "unclaimed": unclaimed}


def by_kind(rows: list[dict]) -> dict[str, dict]:
    agg: dict[str, dict] = {}
    for r in rows:
        a = agg.setdefault(r["kind"], {"bytes": 0.0, "ms": 0.0,
                                       "floor_ms": 0.0, "excess_ms": 0.0})
        a["bytes"] += r["bytes"]
        a["ms"] += r["ms"]
        a["floor_ms"] += r.get("floor_ms", 0.0)
        a["excess_ms"] += r.get("excess_ms", 0.0)
    for a in agg.values():
        a["gbs"] = a["bytes"] / (a["ms"] * 1e6) if a["ms"] > 0 else None
    return agg


def _fmt_gbs(v):
    return f"{v:8.1f}" if v is not None else "       -"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("artifact", help="the artifact the timeline was run on")
    ap.add_argument("--timeline", required=True,
                    help="a decode-timeline --json-out file")
    ap.add_argument("--peak-gbs", type=float, default=None,
                    help="this box's peak memory bandwidth (M4 Max 546, "
                         "M3 Ultra 819). Without it there is no floor and no "
                         "excess column, only GB/s")
    ap.add_argument("--gather-rows", type=int, default=8)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    art = Path(a.artifact)
    tl = json.loads(Path(a.timeline).read_text())
    if Path(tl.get("artifact", "")).name != art.name:
        print(f"WARNING: the timeline was run on {tl.get('artifact')!r}, not "
              f"{art.name!r}. A pinned copy under another name is fine; a "
              "different artifact is not.", file=sys.stderr)
    byts = bytes_by_stage(art, a.gather_rows)
    j = join(tl, byts, a.peak_gbs)
    rows, kinds = j["rows"], by_kind(j["rows"])
    whole = float(tl.get("whole_ms") or sum(r["ms"] for r in rows))
    total_b = sum(r["bytes"] for r in rows)
    noise = float(tl.get("noise_floor_ms") or 0.0)

    if a.json:
        print(json.dumps({"artifact": str(art), "timeline": a.timeline,
                          "context": tl.get("context"), "whole_ms": whole,
                          "peak_gbs": a.peak_gbs, "bytes_per_token": total_b,
                          "noise_floor_ms": noise, "by_kind": kinds,
                          "stages": rows, "unclaimed_bytes": j["unclaimed"]},
                         indent=1))
        return 0

    print(f"\nstage-bandwidth  {art.name}   (timeline context "
          f"{tl.get('context')} tokens, reps {tl.get('reps')})")
    print(f"  whole step {whole:.2f} ms, {total_b/1e9:.3f} GB/token of "
          f"weights -> {total_b/(whole*1e6):.1f} GB/s overall", end="")
    if a.peak_gbs:
        floor = total_b / (a.peak_gbs * 1e6)
        print(f"  ({100*floor/whole:.1f}% of {a.peak_gbs:.0f} GB/s; "
              f"roofline {floor:.2f} ms)")
    else:
        print()
    print(f"  timeline verdict: {tl.get('verdict', '?')}")
    print(f"  noise floor {noise:.3f} ms: no single stage below it is "
          "measured, only aggregates are\n")

    head = f"  {'kind':<12} {'GB/token':>9} {'ms':>8} {'GB/s':>8}"
    if a.peak_gbs:
        head += f" {'floor ms':>9} {'excess ms':>10} {'excess%':>8}"
    print(head)
    tot_excess = sum(k.get("excess_ms", 0.0) for k in kinds.values()) or 1.0
    order = sorted(kinds.items(),
                   key=lambda kv: -(kv[1]["excess_ms"] if a.peak_gbs else kv[1]["ms"]))
    for name, k in order:
        line = (f"  {name:<12} {k['bytes']/1e9:>9.4f} {k['ms']:>8.2f} "
                f"{_fmt_gbs(k['gbs'])}")
        if a.peak_gbs:
            line += (f" {k['floor_ms']:>9.2f} {k['excess_ms']:>10.2f} "
                     f"{100*k['excess_ms']/tot_excess:>7.1f}%")
        print(line)

    key = "excess_ms" if a.peak_gbs else "ms"
    print(f"\n  TOP {a.top} STAGES BY {'EXCESS' if a.peak_gbs else 'TIME'}"
          " (time above the roofline is the waste to chase)")
    for r in sorted(rows, key=lambda r: -r.get(key, 0.0))[: a.top]:
        flag = "  < noise floor" if abs(r["ms"]) < noise else ""
        line = (f"    {r['name']:<14} {r['bytes']/1e6:>9.2f} MB {r['ms']:>8.3f} ms "
                f"{_fmt_gbs(r['gbs'])} GB/s")
        if a.peak_gbs:
            line += f"  excess {r['excess_ms']:>7.3f} ms"
        print(line + flag)
    if j["unclaimed"]:
        print("\n  BYTES NO TIMED STAGE CLAIMS (a layer the timeline does not "
              "run, or a tensor name this tool does not place):")
        for k, v in sorted(j["unclaimed"].items(), key=lambda kv: -kv[1])[:10]:
            print(f"    {k:<14} {v/1e6:>9.2f} MB")
    print("\n  KV-cache reads are NOT in the bytes column (weights only): "
          "attention GB/s understate their traffic at long context.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
