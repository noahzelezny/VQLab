#!/usr/bin/env python3
"""397B v2 mixed-codebook sweep driver — splice ONE expert layer, score, repeat.

WHAT THIS IS. The GLM ledger's "BEST-8 BY MEASURED EFFECT" recipe
(research/glm53-flash/LEDGER.md:411-437) ported to Qwen3.5-397B-A17B. It
promotes a single expert layer from the 2.4bpw base's d4/K256 codes to a
richer donor geometry (d4/K512 from the shipped 2.6bpw rung, or d4/K2048
from the 3.1bpw rung), scores the ASSEMBLED model, and tabulates the
measured single-layer effect. Layers are then chosen by that measurement,
never by `vqlab layer-leverage` — that probe was falsified for MoE
allocation on GLM (LEDGER.md:277-289) and its top-8 contained three
actively harmful layers (LEDGER.md:391-394).

WHY A NEW SPLICE TOOL. The GLM sweep's `scratchpad/glm_mix.py` and
`scratchpad/glm_layer_sweep.sh` were never committed and are gone from
disk. Only their outputs survive. The splice contract they implemented is
documented at research/allocation/METHOD.md:193-199 (weight_map equality
assertion; per-key lookup + index rebuild when layouts differ). Here the
assertion HOLDS: the 2.4 / 2.6 / 3.1 artifacts share all 2545 keys AND
their shard assignment, so a promotion rewrites only the 1-2 shards that
carry the layer and hardlinks the other 25-26.

SAFETY. Nothing here loads a model. Scoring is delegated to
referee/score_streaming.py, which streams blocks by design and is exempt
from the resident-memory rule (preflight_ram.py:16). The preflight below
guards the WORKING SET this driver actually allocates (one shard in
memory for the rewrite) plus the scorer's flat footprint, and refuses to
start if it cannot fit — in the spirit of preflight_ram.py, applied to
the thing this script is actually responsible for.

USAGE

  # validate everything, load nothing, print the plan
  ./v2_sweep.py --dry-run

  # run the sweep (resumable; Ctrl-C safe, re-run to continue)
  ./v2_sweep.py --layers 0-56

  # build a named multi-layer candidate (stage 3)
  ./v2_sweep.py --combo 12,17,23,29,31,40,44,51 --tag best8

STATE. --state (default state.json) lists completed candidates and is
consulted on every start; a completed candidate is skipped. Per-candidate
rows are appended to --tsv (default sweep.tsv) in the order they finish.
"""

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

GiB = 2 ** 30

E_DEFAULT = "/Volumes/Thunderbay SSD/Exo Models"
BASE_DEFAULT = "TheDrainFlorist--Qwen3.5-397B-A17B-VQ-2.4bpw"
DONORS = {
    # name -> (artifact dir, k, pack_bits, GiB added per promoted layer)
    # Byte costs are DERIVED, not assumed: (index total_size donor - base)
    # / 57 layers. Verified 2026-09-03 against the published sizes --
    # 57 x 0.1875 = 10.69 GiB takes 111.62 -> 122.30 (the 2.6bpw card's
    # own figure) and 57 x 0.5625 = 32.06 takes it to 143.68 (the 3.1bpw
    # card's). Both land exactly.
    "k512": ("TheDrainFlorist--Qwen3.5-397B-A17B-VQ-2.6bpw", 512, 9, 0.1875),
    "k2048": ("TheDrainFlorist--Qwen3.5-397B-A17B-VQ-3.1bpw", 2048, 11, 0.5625),
}
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
VQ_SUFFIXES = ("codebook", "codes", "vq_scales")
N_VQ_LAYERS = 57            # vq_modules covers layers 0-56, no gaps

# Working set this driver allocates: the largest single shard it rewrites,
# held as arrays while the replacement shard is written, plus the scorer's
# measured flat footprint (score_streaming.py docstring: "~15G").
SCORER_FLAT_GIB = 15.0


def die(msg, code=2):
    print(f"PREFLIGHT FAIL: {msg}", file=sys.stderr)
    sys.exit(code)


def module_name(layer, proj):
    return f"language_model.model.layers.{layer}.mlp.switch_mlp.{proj}"


def load_artifact(path):
    p = pathlib.Path(path)
    idx = json.load(open(p / "model.safetensors.index.json"))
    cfg = json.load(open(p / "config.json"))
    return p, idx, cfg


def parse_layers(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def physical_ram_gib():
    try:
        n = subprocess.run(["sysctl", "-n", "hw.memsize"],
                           capture_output=True, text=True, check=True)
        return int(n.stdout.strip()) / GiB
    except Exception:
        return None


# --------------------------------------------------------------------------
# preflight — refuse to start rather than thrash

def preflight(base, donor, args, layers):
    bp, bidx, bcfg = load_artifact(base)
    dp, didx, dcfg = load_artifact(donor)
    bwm, dwm = bidx["weight_map"], didx["weight_map"]

    if set(bwm) != set(dwm):
        die(f"key sets differ: base {len(bwm)} vs donor {len(dwm)} tensors. "
            "The splice contract (METHOD.md:193-199) needs a per-key lookup "
            "and an index rebuild before this can run.")
    mismatched = [k for k in bwm if bwm[k] != dwm[k]]
    if mismatched:
        die(f"{len(mismatched)} tensors sit in different shards between base "
            f"and donor (e.g. {mismatched[0]}). Shard-level hardlinking is "
            "invalid; relax to a per-key rewrite first (METHOD.md:193-199).")

    bvm, dvm = bcfg.get("vq_modules"), dcfg.get("vq_modules")
    if not bvm or not dvm:
        die("one of the artifacts has no vq_modules block in config.json")
    if set(bvm) != set(dvm):
        die("vq_modules coverage differs between base and donor")

    # every module must be flat at the expected geometry on both sides
    bk = {v["k"] for v in bvm.values()}
    dk = {v["k"] for v in dvm.values()}
    bd = {v["dim"] for v in bvm.values()}
    dd = {v["dim"] for v in dvm.values()}
    if len(bk) != 1 or len(dk) != 1:
        die(f"non-flat geometry: base k={sorted(bk)} donor k={sorted(dk)}. "
            "This driver assumes a flat base; a mixed base needs the "
            "per-module levels tracked in the state file.")
    if bd != dd:
        die(f"dim differs: base d={sorted(bd)} donor d={sorted(dd)}. A "
            "promotion must change K only; changing d changes the "
            "reconstruction, not just the codebook size.")
    if dk <= bk:
        die(f"donor K {sorted(dk)} is not richer than base K {sorted(bk)}")

    for layer in layers:
        for proj in PROJECTIONS:
            m = module_name(layer, proj)
            if m not in bvm:
                die(f"layer {layer}: {m} is not a VQ module in the base")

    # byte cost, derived from the two indexes rather than assumed
    b_tot = bidx.get("metadata", {}).get("total_size")
    d_tot = didx.get("metadata", {}).get("total_size")
    per_layer_gib = (d_tot - b_tot) / N_VQ_LAYERS / GiB

    # shards touched, and the largest one -- that is the working set
    touched = {}
    for layer in layers:
        s = set()
        for proj in PROJECTIONS:
            for suf in VQ_SUFFIXES:
                s.add(bwm[f"{module_name(layer, proj)}.{suf}"])
        touched[layer] = sorted(s)
    biggest = max(
        sum(os.path.getsize(bp / f) for f in sh) for sh in touched.values()
    ) / GiB

    # A rewrite holds the shard's tensors in memory while writing the new
    # one: budget 2x the shard (read side + write side) plus slack.
    need = biggest * 2 + 2.0
    ram = physical_ram_gib()
    if ram is not None:
        bar = ram * args.headroom
        print(f"PREFLIGHT  splice working set {need:.2f} GiB "
              f"(largest shard group {biggest:.2f} GiB x2 + 2.0)   "
              f"box RAM {ram:.0f} GiB   bar {bar:.1f} GiB "
              f"({args.headroom:.0%})   -> {'OK' if need <= bar else 'FAIL'}")
        if need > bar:
            die(f"the splice working set ({need:.2f} GiB) does not fit under "
                f"{bar:.1f} GiB on this box. Run on a bigger box or reduce "
                "the shard group.")
        print(f"PREFLIGHT  scorer is streaming (score_streaming.py, flat "
              f"~{SCORER_FLAT_GIB:.0f} GiB) and is exempt from the resident "
              "rule, per preflight_ram.py:16")
    else:
        print("PREFLIGHT  WARN: could not read hw.memsize; RAM bar not checked")

    # disk: one candidate at a time unless --keep
    st = os.statvfs(args.workdir if os.path.isdir(args.workdir) else E_DEFAULT)
    free = st.f_bavail * st.f_frsize / GiB
    concurrent = len(layers) if args.keep else 1
    disk_need = biggest * 1.05 * concurrent
    print(f"PREFLIGHT  disk need {disk_need:.1f} GiB "
          f"({'all candidates kept' if args.keep else 'one at a time'})   "
          f"free {free:.1f} GiB   -> {'OK' if disk_need <= free else 'FAIL'}")
    if disk_need > free:
        die(f"need {disk_need:.1f} GiB of scratch and only {free:.1f} GiB is "
            "free. Drop --keep, or free space.")

    # instruments must exist before anything runs
    for label, path in (("scorer", args.scorer), ("corpus", args.corpus)):
        if not os.path.exists(path):
            die(f"{label} not found: {path}")
    if args.corpus_code and not os.path.exists(args.corpus_code):
        die(f"code corpus not found: {args.corpus_code}")
    if not os.path.exists(args.python):
        die(f"interpreter not found: {args.python}")

    print(f"PREFLIGHT  base   {bp.name}  {b_tot / GiB:.3f} GiB")
    print(f"PREFLIGHT  donor  {dp.name}  {d_tot / GiB:.3f} GiB  "
          f"K{sorted(dk)[0]}")
    print(f"PREFLIGHT  promotion cost {per_layer_gib:.4f} GiB/layer  "
          f"({len(layers)} candidates queued)")
    print("PREFLIGHT  OK")
    return dict(bp=bp, dp=dp, bwm=bwm, bvm=bvm, dvm=dvm,
                per_layer_gib=per_layer_gib, touched=touched,
                b_tot=b_tot, biggest=biggest)


# --------------------------------------------------------------------------
# splice

def splice(pf, layers, out_dir, verbose=True):
    """Build an artifact = base with `layers` promoted to the donor geometry.

    Untouched shards are HARDLINKED (no copy, no extra bytes). Touched
    shards are rewritten with the donor's codebook/codes/vq_scales for the
    promoted modules and the base's tensors for everything else.
    """
    import mlx.core as mx
    # CPU-only, for the same reason splice_ple.py:18 does it: this is a pure
    # byte transform over shards that may sit on an external/SMB volume, and
    # a GPU command buffer around a slow read trips the Metal watchdog.
    # EXPERIMENTS.md:4727 is the 397B instance of exactly that failure.
    mx.set_default_device(mx.cpu)

    bp, dp = pf["bp"], pf["dp"]
    out = pathlib.Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    promoted = set()
    touched = set()
    for layer in layers:
        for proj in PROJECTIONS:
            promoted.add(module_name(layer, proj))
        touched.update(pf["touched"][layer])

    # 1. every non-shard file comes across as-is
    for f in bp.iterdir():
        if f.suffix == ".safetensors" or f.is_dir():
            continue
        shutil.copy2(f, out / f.name)

    # 2. shards
    all_shards = sorted(set(pf["bwm"].values()))
    for sh in all_shards:
        if sh not in touched:
            os.link(bp / sh, out / sh)
            continue
        base_t = mx.load(str(bp / sh))
        donor_t = mx.load(str(dp / sh))
        n_swapped = 0
        for name in list(base_t):
            mod, _, suf = name.rpartition(".")
            if mod in promoted and suf in VQ_SUFFIXES:
                base_t[name] = donor_t[name]
                n_swapped += 1
        # mx.save_safetensors FORCES a .safetensors extension -- a ".tmp"
        # suffix is silently rewritten to ".tmp.safetensors" and the swap
        # then fails on a missing file. Keep the extension last.
        tmp = out / (sh[:-len(".safetensors")] + ".tmp.safetensors")
        mx.save_safetensors(str(tmp), base_t)
        os.replace(tmp, out / sh)
        del base_t, donor_t
        if verbose:
            print(f"    rewrote {sh}: {n_swapped} tensors swapped", flush=True)

    # 3. config.json — promoted modules take the donor's vq_modules entry
    cfg = json.load(open(out / "config.json"))
    for mod in promoted:
        cfg["vq_modules"][mod] = pf["dvm"][mod]
    json.dump(cfg, open(out / "config.json", "w"), indent=1)

    # 4. index total_size must match the bytes on disk, or the loader and
    #    every downstream size claim disagree with the artifact.
    idx = json.load(open(out / "model.safetensors.index.json"))
    idx.setdefault("metadata", {})["total_size"] = sum(
        os.path.getsize(out / s) for s in all_shards)
    json.dump(idx, open(out / "model.safetensors.index.json", "w"), indent=1)

    size = idx["metadata"]["total_size"] / GiB
    if verbose:
        print(f"    spliced {out.name}: {size:.3f} GiB", flush=True)
    return size


# --------------------------------------------------------------------------
# score

def score(args, model_dir, corpus):
    cmd = [args.python, args.scorer, "--model", str(model_dir),
           "--corpus", corpus, "--max-tokens", str(args.tokens)]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-3000:], file=sys.stderr)
        print(p.stderr[-3000:], file=sys.stderr)
        raise RuntimeError(f"scorer failed rc={p.returncode}")
    rec = None
    for line in p.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                pass
    if rec is None or "ppl" not in rec:
        print(p.stdout[-3000:], file=sys.stderr)
        raise RuntimeError("scorer produced no result record")
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec


# --------------------------------------------------------------------------

def load_state(path):
    if os.path.exists(path):
        return json.load(open(path))
    return {"done": {}, "base": None}


def save_state(path, st):
    tmp = path + ".tmp"
    json.dump(st, open(tmp, "w"), indent=1)
    os.replace(tmp, path)


def append_tsv(path, row, header):
    new = not os.path.exists(path)
    with open(path, "a") as fh:
        if new:
            fh.write("\t".join(header) + "\n")
        fh.write("\t".join(str(row.get(h, "")) for h in header) + "\n")


TSV_HEADER = ["candidate", "layers", "donor", "gib", "d_gib",
              "prose_ppl", "d_prose", "code_ppl", "d_code",
              "tokens", "wall_s", "utc"]


def main():
    ap = argparse.ArgumentParser(
        description="397B v2 mixed-codebook single-layer sweep")
    ap.add_argument("--models-root", default=E_DEFAULT)
    ap.add_argument("--base", default=BASE_DEFAULT,
                    help="base artifact dir name under --models-root")
    ap.add_argument("--donor", default="k512", choices=sorted(DONORS),
                    help="richer geometry to promote INTO (default k512, "
                         "the 2.6bpw rung: +0.1875 GiB/layer)")
    ap.add_argument("--layers", default="0-56",
                    help="single-layer candidates to sweep, e.g. 0-56 or 3,7,9")
    ap.add_argument("--combo", default=None,
                    help="build ONE candidate promoting all these layers "
                         "(stage 3); implies a single run")
    ap.add_argument("--tag", default=None, help="name for a --combo candidate")
    ap.add_argument("--workdir", default=None,
                    help="where candidates are built (default <models-root>"
                         "/v2sweep397b)")
    ap.add_argument("--state", default=None)
    ap.add_argument("--tsv", default=None)
    ap.add_argument("--tokens", type=int, default=8192,
                    help="referee prefix length. 8192 is the published "
                         "instrument; a shorter prefix ranks faster but its "
                         "numbers are NOT comparable to the cards.")
    ap.add_argument("--code", action="store_true",
                    help="also score the code corpus (roughly doubles cost; "
                         "prefer this only on finalists)")
    ap.add_argument("--keep", action="store_true",
                    help="keep every candidate on disk instead of deleting "
                         "after scoring")
    ap.add_argument("--budget-gib", type=float, default=1.5,
                    help="promotion budget; the driver WARNS when a combo "
                         "exceeds it (see V2-SWEEP-PLAN.md Q1)")
    ap.add_argument("--headroom", type=float, default=0.90,
                    help="fraction of RAM the splice working set may take")
    ap.add_argument("--max-candidates", type=int, default=None,
                    help="stop after this many NEW candidates this run "
                         "(use to fit a night)")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter for the referee. MUST be a STOCK "
                         "mlx-lm environment (the published 397B numbers "
                         "were all measured on one). On the M3 that is "
                         "/opt/anaconda3/bin/python3 (mlx_lm 0.31.3). NEVER "
                         "point this at an exo env "
                         "(/opt/anaconda3/envs/exo, "
                         "/opt/homebrew/anaconda3/envs/exo): those carry a "
                         "grafted mlx-lm and a jaccl mlx fork, so any number "
                         "from them is off-instrument.")
    ap.add_argument("--scorer", default=None)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--corpus-code", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="validate paths, geometry and budget; load nothing")
    a = ap.parse_args()

    here = pathlib.Path(__file__).resolve().parent
    ql = here.parent.parent            # research/quantlab
    a.scorer = a.scorer or str(ql / "referee" / "score_streaming.py")
    a.corpus = a.corpus or str(ql / "referee" / "referee_corpus.txt")
    a.corpus_code = a.corpus_code or str(ql / "referee" / "referee_corpus_code.txt")
    a.workdir = a.workdir or os.path.join(a.models_root, "v2sweep397b")
    a.state = a.state or str(here / "state.json")
    a.tsv = a.tsv or str(here / "sweep.tsv")

    base = os.path.join(a.models_root, a.base)
    donor_name, dk, dpb, _ = DONORS[a.donor]
    donor = os.path.join(a.models_root, donor_name)
    for p in (base, donor):
        if not os.path.isdir(p):
            die(f"artifact not found: {p}")

    if a.combo:
        combos = [(a.tag or "combo_" + "_".join(a.combo.split(",")),
                   parse_layers(a.combo))]
    else:
        combos = [(f"L{l}", [l]) for l in parse_layers(a.layers)]

    all_layers = sorted({l for _, ls in combos for l in ls})
    pf = preflight(base, donor, a, all_layers)

    for tag, layers in combos:
        cost = len(layers) * pf["per_layer_gib"]
        if cost > a.budget_gib + 1e-6:      # exactly-on-budget is not over
            print(f"WARNING  {tag}: {len(layers)} layers = +{cost:.3f} GiB, "
                  f"OVER the {a.budget_gib:.2f} GiB budget "
                  "(see V2-SWEEP-PLAN.md Q1)")

    if a.dry_run:
        print(f"\nDRY RUN — nothing loaded, nothing built.")
        print(f"  workdir     {a.workdir}")
        print(f"  state       {a.state}")
        print(f"  tsv         {a.tsv}")
        print(f"  scorer      {a.scorer} --max-tokens {a.tokens}")
        print(f"  corpus      {a.corpus}")
        if a.code:
            print(f"  code corpus {a.corpus_code}")
        print(f"  candidates  {len(combos)}  "
              f"(first {[t for t, _ in combos[:5]]})")
        print(f"  each promotes to K{dk} (pack_bits {dpb}) at "
              f"+{pf['per_layer_gib']:.4f} GiB/layer")
        print("  per candidate: rewrite "
              f"{pf['biggest']:.2f} GiB of shards, hardlink the rest")
        st = load_state(a.state)
        print(f"  already done: {len(st['done'])}")
        return

    os.makedirs(a.workdir, exist_ok=True)
    st = load_state(a.state)
    st["base"] = a.base
    st.setdefault("donor", a.donor)
    if st["donor"] != a.donor:
        die(f"state file was built against donor {st['donor']}, not "
            f"{a.donor}. Layer effects are base- and donor-specific "
            "(LEDGER.md:460-478); use a separate --state.")

    # baseline score, once
    if "BASE" not in st["done"]:
        print("\n=== scoring the base (reference row) ===", flush=True)
        rec = score(a, base, a.corpus)
        row = {"candidate": "BASE", "layers": "", "donor": a.donor,
               "gib": round(pf["b_tot"] / GiB, 3), "d_gib": 0.0,
               "prose_ppl": rec["ppl"], "d_prose": 0.0,
               "tokens": a.tokens, "wall_s": rec["wall_s"],
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if a.code:
            rc = score(a, base, a.corpus_code)
            row["code_ppl"] = rc["ppl"]
            row["d_code"] = 0.0
            row["wall_s"] += rc["wall_s"]
        st["done"]["BASE"] = row
        save_state(a.state, st)
        append_tsv(a.tsv, row, TSV_HEADER)
        print(f"  BASE prose ppl {row['prose_ppl']}  "
              f"({row['wall_s']}s)", flush=True)

    base_prose = st["done"]["BASE"]["prose_ppl"]
    base_code = st["done"]["BASE"].get("code_ppl")

    n_new = 0
    for tag, layers in combos:
        if tag in st["done"]:
            continue
        if a.max_candidates is not None and n_new >= a.max_candidates:
            print(f"\nstopping: --max-candidates {a.max_candidates} reached")
            break
        print(f"\n=== {tag}  (layers {layers}) ===", flush=True)
        t0 = time.time()
        cand = os.path.join(a.workdir, f"397b-v2-{tag}")
        try:
            size = splice(pf, layers, cand)
            rec = score(a, cand, a.corpus)
            row = {"candidate": tag, "layers": ",".join(map(str, layers)),
                   "donor": a.donor, "gib": round(size, 3),
                   "d_gib": round(size - pf["b_tot"] / GiB, 4),
                   "prose_ppl": rec["ppl"],
                   "d_prose": round(base_prose - rec["ppl"], 6),
                   "tokens": a.tokens,
                   "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            if a.code:
                rc = score(a, cand, a.corpus_code)
                row["code_ppl"] = rc["ppl"]
                if base_code is not None:
                    row["d_code"] = round(base_code - rc["ppl"], 6)
            row["wall_s"] = round(time.time() - t0, 1)
        finally:
            if not a.keep and os.path.isdir(cand):
                shutil.rmtree(cand, ignore_errors=True)

        st["done"][tag] = row
        save_state(a.state, st)
        append_tsv(a.tsv, row, TSV_HEADER)
        n_new += 1
        done = len([k for k in st["done"] if k != "BASE"])
        rate = row["wall_s"]
        print(f"  {tag}: prose {row['prose_ppl']}  d_prose "
              f"{row['d_prose']:+.4f}  {rate:.0f}s   "
              f"[{done}/{len(combos)} done, ~"
              f"{(len(combos) - done) * rate / 3600:.1f}h left at this rate]",
              flush=True)

    print(f"\nwrote {a.tsv} and {a.state}")


if __name__ == "__main__":
    main()
