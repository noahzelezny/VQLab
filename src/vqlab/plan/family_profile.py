#!/usr/bin/env python3
"""vqlab family-profile — size up a teacher from its headers, before any fit.

Onboarding step 1 (docs/ONBOARDING.md), mechanized. Reads ONLY config.json,
the index and each shard's safetensors header -- no tensor data, no GPU, a
few seconds even for the 751 GB 397B teacher, safe on a busy box -- and
writes a machine-readable profile other tools read instead of a hand-kept
table:

    families/<family>/teachers/<teacher>/profile.json

  identity    teacher slug, path, HF repo@revision when it is a hub snapshot,
              config sha256, shard fingerprint (records/provenance.py)
  arch        model_type, layers, experts, hidden / expert widths, which
              layers are dense, vision tower, MTP head
  bytes       every tensor classified (experts / carry / embed / vision /
              mtp) at its source dtype, and expert bytes PER LAYER
  modules     one row per expert projection: (E, OUT, IN) -- the shape
              signature fit parts are attributed by (core/fitstore.py)
  geometries  per projection, which (d, K) are LEGAL: exact packing
              (nsub % 32 == 0, F97) and the fused-kernel threadgroup ceiling
              (K*d*2 < 32768, FINDINGS IV) -- and their exact packed GiB
              for the whole expert stack, so a byte budget can be priced by
              arithmetic before anything is fit

    vqlab family-profile --teacher <dir> [--family NAME] [--out DIR]
    vqlab family-profile --teacher <dir> --suggest-entry   # unknown family

An unknown architecture is not an error: the profile is still written from
the header census, and --suggest-entry prints a draft families.py entry
inferred from the index for a human to confirm. Measured facts (init
verdict, depth knee, safe K band) are appended to the same folder by the
tools that measure them; this tool only writes what headers can prove.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import pathlib
import re
import struct
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # src/
from vqlab import _layout  # noqa: E402,F401  one module object per name
from vqlab import config  # noqa: E402
from families import DENSE_FAMILIES, FAMILY  # noqa: E402
import fitstore  # noqa: E402
import provenance  # noqa: E402

REPO = _layout.SRC.parent
GSZ = 64                       # scale group, every shipped geometry
TG_BYTES = 32768               # threadgroup codebook cache ceiling
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "F8_E5M2": 1,
               "I8": 1, "U8": 1, "I32": 4, "U32": 4, "I64": 8}
DIMS = (2, 4, 8)
KS = tuple(2 ** b for b in range(4, 17))          # K16 .. K65536


def header(p):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h


def slug(teacher: pathlib.Path, fp: dict) -> str:
    return fitstore.teacher_slug(teacher)      # one naming rule, store + profiles


def resolve_family(cfg: dict, want: str | None):
    if want:
        return want if want in FAMILY else None
    types = {t for t in (cfg.get("model_type"),
                         (cfg.get("text_config") or {}).get("model_type")) if t}
    for name, spec in FAMILY.items():
        if spec.get("model_type") in types or name in types:
            return name
    for name in FAMILY:                               # qwen3_5_moe -> qwen3_5
        if any(t.startswith(name + "_") for t in types):
            return name
    return None


def resolve_dense(shapes):
    """A dense teacher: the DENSE_FAMILIES entry whose MLP key template is
    present in the index. Returns (name, template) or (None, None)."""
    for name, (tmpl, _rng) in DENSE_FAMILIES.items():
        if tmpl.format(li=0, key="gate_proj") in shapes:
            return name, tmpl
    return None, None


def dense_rows(tmpl, shapes, n_layers):
    rows = []
    for li in range(n_layers):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            shp = shapes.get(tmpl.format(li=li, key=proj))
            if shp is not None:
                OUT, IN = shp
                rows.append({"layer": li, "proj": proj, "E": 1, "OUT": OUT, "IN": IN})
    return rows


def classify(key: str, expert_rx) -> str:
    k = key.lower()
    if "vision" in k or "visual" in k:
        return "vision"
    if ".mtp." in k or k.startswith("mtp."):
        return "mtp"
    if expert_rx and expert_rx.search(key):
        return "vq_target"
    if "embed_tokens" in k or "lm_head" in k:
        return "embed"
    return "carry"


def expert_regex(spec, dense_tmpl=None):
    """Regex over SOURCE keys for the tensors VQ replaces: routed experts
    (MoE) or the MLP trio (dense)."""
    if dense_tmpl:
        pat = re.escape(dense_tmpl).replace("\\{li\\}", r"\d+").replace(
            "\\{key\\}", r"[a-z_]+")
        return re.compile("^" + pat + "$")
    if not spec:
        return None
    pat = re.escape(spec["src_key"])
    for tok in ("\\{li\\}", "\\{e\\}"):
        pat = pat.replace(tok, r"\d+")
    pat = pat.replace("\\{key\\}", r"[a-z_]+")
    return re.compile("^" + pat + r"(\.weight)?$")


def module_rows(spec, idx_shapes, n_layers):
    """(layer, proj) -> (E, OUT, IN) for every expert projection present."""
    rows = []
    for li in range(n_layers):
        for proj, (src, half) in spec["proj"].items():
            base = spec["src_key"].format(li=li, key=src, e=0)
            shape = idx_shapes.get(base) or idx_shapes.get(base + ".weight")
            if shape is None:
                continue
            if "{e}" in spec["src_key"]:              # unfused per-expert 2D
                rx = re.compile(re.escape(spec["src_key"].format(
                    li=li, key=src, e="@")).replace("@", r"\d+") + r"(\.weight)?$")
                E = sum(1 for k in idx_shapes if rx.match(k))
                OUT, IN = shape
            else:
                E, OUT, IN = shape
            if half is not None:
                OUT //= 2
            rows.append({"layer": li, "proj": proj, "E": E, "OUT": OUT, "IN": IN})
    return rows


def geometry_table(IN: int, E: int, OUT: int):
    """Every (d, K) with its legality and exact packed bytes for ONE module."""
    out = []
    for d in DIMS:
        if IN % d:
            continue
        nsub = IN // d
        for K in KS:
            bits = int(math.log2(K))
            exact = nsub % 32 == 0
            fused = K * d * 2 < TG_BYTES
            words = math.ceil(nsub / 32) * bits
            codes = E * OUT * words * 4
            scales = E * OUT * (IN // GSZ) * 2
            cb = K * d * 2
            out.append({"d": d, "K": K, "bits_per_weight": round(bits / d + 16 / GSZ, 4),
                        "exact_packing": exact, "fused_kernel_ok": fused,
                        "legal": exact and fused, "bytes": codes + scales + cb})
    return out


def suggest_entry(idx_keys):
    """Draft a FAMILY entry by pattern-matching expert-looking keys."""
    cands, suffixed = {}, {}
    for k in idx_keys:
        if "expert" not in k or "shared" in k:
            continue
        t = re.sub(r"experts\.\d+\.", "experts.{e}.", k)
        t = re.sub(r"\.\d+\.", ".{li}.", t, count=1)   # first index = layer
        t = re.sub(r"\.weight$", "", t)
        head, _, proj = t.rpartition(".")
        cands.setdefault(head + ".{key}", set()).add(proj)
        # keep the key's real spelling: a template without ".weight" on a
        # checkpoint that has it matches nothing a loader can open
        suffixed[head + ".{key}"] = k.endswith(".weight")
    if not cands:
        return None
    tmpl, projs = max(cands.items(), key=lambda kv: len(kv[1]))
    if suffixed.get(tmpl):
        tmpl += ".weight"
    proj_map = {}
    if "gate_up_proj" in projs:
        proj_map = {"gate_proj": ("gate_up_proj", 0), "up_proj": ("gate_up_proj", 1)}
    for p in sorted(projs):
        if p != "gate_up_proj":
            proj_map[p] = (p, None)
    return {"kind": "moe", "target_substr": "switch_mlp", "src_key": tmpl,
            "proj": proj_map,
            "_note": "DRAFT from index key patterns: confirm target_substr "
                     "against the runtime's module tree, then run "
                     "`python src/vqlab/core/expert_src.py --selftest`"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vqlab family-profile",
                                 description=__doc__.split("\n")[0])
    ap.add_argument("--teacher", required=True, help="bf16 teacher dir or HF snapshot")
    ap.add_argument("--family", help="families.py name (default: from model_type)")
    ap.add_argument("--out", help="default: <repo>/families/<family>/teachers/<teacher>/")
    ap.add_argument("--suggest-entry", action="store_true",
                    help="print a draft families.py entry inferred from the index")
    ap.add_argument("--print", action="store_true", help="print the profile JSON")
    ap.add_argument("--write-entry", metavar="NAME",
                    help="unknown family: save the suggested entry as "
                         "families/NAME/entry.json (a DATA entry every tool "
                         "reads; no code change), then re-profile under it")
    a = ap.parse_args(argv)

    T = pathlib.Path(a.teacher)
    cfg = json.loads((T / "config.json").read_text())
    tc = cfg.get("text_config") or cfg
    shapes, dtypes, by_class = {}, {}, {}
    for shard in sorted(T.glob("*.safetensors")):
        for k, v in header(os.path.realpath(shard)).items():
            shapes[k], dtypes[k] = v["shape"], v["dtype"]
    moe = bool(tc.get("num_experts") or tc.get("n_routed_experts"))
    dense_tmpl = None
    if moe or (a.family and a.family in FAMILY):
        fam = resolve_family(cfg, a.family)
        spec = FAMILY.get(fam) if fam else None
    else:
        fam, dense_tmpl = resolve_dense(shapes)
        if a.family:
            fam, dense_tmpl = a.family, (DENSE_FAMILIES.get(a.family) or (None,))[0]
        spec = None
    rx = expert_regex(spec, dense_tmpl)
    per_layer = {}
    for k, shp in shapes.items():
        n = math.prod(shp) * DTYPE_BYTES.get(dtypes[k], 2)
        c = classify(k, rx)
        by_class[c] = by_class.get(c, 0) + n
        if c == "vq_target":
            m = re.search(r"layers\.(\d+)\.", k)
            if m:
                per_layer[int(m.group(1))] = per_layer.get(int(m.group(1)), 0) + n

    L = tc.get("num_hidden_layers") or 0
    rows = (dense_rows(dense_tmpl, shapes, L + 8) if dense_tmpl
            else module_rows(spec, shapes, L + 8) if spec else [])
    fp = provenance.fingerprint(T)
    sl = slug(T, fp)
    sigs = {}
    for r in rows:
        sigs.setdefault(r["proj"], {(r["E"], r["OUT"], r["IN"])})
        sigs[r["proj"]].add((r["E"], r["OUT"], r["IN"]))
    geos = {}
    for proj, ss in sigs.items():
        E, OUT, IN = sorted(ss)[0]
        n_mod = sum(1 for r in rows if r["proj"] == proj)
        geos[proj] = [{**g, "stack_gib": round(g["bytes"] * n_mod / 2 ** 30, 3)}
                      for g in geometry_table(IN, E, OUT)]
    expert_params = sum(r["E"] * r["OUT"] * r["IN"] for r in rows)
    prof = {
        "schema": "vqlab.family-profile/1",
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "family": fam, "teacher": sl,
        "identity": {"path": config.portable(T), "hf_repo": fp.get("hf_repo"),
                     "hf_revision": fp.get("hf_revision"),
                     "config_sha256": hashlib.sha256((T / "config.json").read_bytes()).hexdigest(),
                     "shards": len(fp.get("shards", {})),
                     "shard_fingerprint": hashlib.sha256(json.dumps(
                         fp.get("shards", {}), sort_keys=True).encode()).hexdigest()},
        "arch": {"model_type": cfg.get("model_type"),
                 "text_model_type": (cfg.get("text_config") or {}).get("model_type"),
                 "layers": L,
                 "experts": tc.get("num_experts") or tc.get("n_routed_experts"),
                 "experts_per_token": tc.get("num_experts_per_tok"),
                 "hidden": tc.get("hidden_size"),
                 "expert_intermediate": tc.get("moe_intermediate_size"),
                 "dense_intermediate": tc.get("intermediate_size"),
                 "kind": "moe" if moe else "dense",
                 "vq_layers": sorted({r["layer"] for r in rows}),
                 "vision_tower": by_class.get("vision", 0) > 0,
                 "mtp_head": by_class.get("mtp", 0) > 0},
        "bytes": {"by_class_gib": {k: round(v / 2 ** 30, 3) for k, v in sorted(by_class.items())},
                  "vq_target_gib_per_layer": {str(k): round(v / 2 ** 30, 4)
                                              for k, v in sorted(per_layer.items())},
                  "vq_target_params": expert_params,
                  "vq_target_gib_per_bit": round(expert_params / 8 / 2 ** 30, 4)},
        "modules": {"signatures": {p: sorted(list(s) for s in ss) for p, ss in sigs.items()},
                    "count": len(rows)},
        "geometries": geos,
    }
    if not rows:           # no registry entry, or one that matched nothing
        prof["unknown_family"] = True
        prof["suggested_entry"] = suggest_entry(shapes)
    out = pathlib.Path(a.out) if a.out else (
        pathlib.Path(os.environ.get("VQLAB_FAMILIES_DIR") or REPO / "families")
        / (fam or "_unknown") / "teachers" / sl)
    out.mkdir(parents=True, exist_ok=True)
    pf = out / "profile.json"
    old = json.loads(pf.read_text()) if pf.exists() else None
    if old and {k: v for k, v in old.items() if k != "created"} == \
            {k: v for k, v in prof.items() if k != "created"}:
        prof["created"] = old["created"]            # unchanged: no git churn
    else:
        pf.write_text(json.dumps(prof, indent=1))

    bc = prof["bytes"]["by_class_gib"]
    print(f"{sl}  family={fam or 'UNKNOWN'}  model_type={cfg.get('model_type')}")
    print(f"  {prof['arch']['kind']}: layers {L}, experts {prof['arch']['experts']}, "
          f"VQ-target layers {len(prof['arch']['vq_layers'])}, modules {len(rows)}")
    print("  source GiB: " + ", ".join(f"{k} {v}" for k, v in bc.items()))
    print(f"  VQ target: {prof['bytes']['vq_target_gib_per_bit']} GiB per bit per weight")
    for proj, gs in geos.items():
        ok = [f"d{g['d']}-K{g['K']}" for g in gs if g["legal"]]
        print(f"  {proj:10s} {sigs[proj]}  legal: {len(ok)} "
              f"({', '.join(ok[:6])}{', ...' if len(ok) > 6 else ''})")
    if not rows:
        print("  UNKNOWN FAMILY: no families.py entry matches. Draft entry:")
        print(json.dumps(prof["suggested_entry"], indent=2))
    elif a.suggest_entry:
        print(json.dumps(suggest_entry(shapes), indent=2))
    if a.write_entry and not rows:
        ent = prof.get("suggested_entry")
        if not ent:
            print("  no expert-like tensors found; nothing to write")
            return 1
        ed_dir = pathlib.Path(os.environ.get("VQLAB_FAMILIES_DIR")
                              or REPO / "families") / a.write_entry
        ed_dir.mkdir(parents=True, exist_ok=True)
        ent["_source"] = f"family-profile --suggest-entry on {sl}"
        (ed_dir / "entry.json").write_text(json.dumps(ent, indent=1))
        print(f"  wrote DRAFT entry {ed_dir / 'entry.json'}; re-profiling under it")
        import families as _f
        importlib_reload = __import__("importlib").reload
        importlib_reload(_f)
        FAMILY.update(_f.FAMILY)
        return main([*(argv if argv is not None else sys.argv[1:]),
                     "--family", a.write_entry][:])  # noqa: E501
    if a.print:
        print(json.dumps(prof, indent=1))
    print(f"-> {out / 'profile.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
