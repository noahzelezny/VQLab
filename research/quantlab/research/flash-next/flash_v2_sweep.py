#!/usr/bin/env python3
"""Flash-Next v2 mixed-codebook sweep driver — promote ONE expert layer, score, repeat.

WHAT THIS IS. The GLM ledger's "BEST-N BY MEASURED EFFECT" recipe applied to
Qwen3.8-Flash-Next. It promotes a single expert layer from the FLAT 2.1bpw
base's d8/K16384 codes to the d2/K256 donor, scores the assembled model with
`vqlab.stream_score` (KL against the cached bf16 teacher), and tabulates the
measured single-layer effect. Layers are chosen by that measurement, never by
`vqlab layer-leverage` -- that probe was falsified for MoE allocation on GLM
(research/glm53-flash/LEDGER.md:265-271, 391-394), and half of it is already
falsified on Flash too: downgrading its 10 quietest layers scored WORSE than
flat (LEDGER.md:267-282).

THE POINT IS THE PROBE VERDICT. The shipped 2.1bpw rung is the probe's hot-2
(L0,L1) and is byte-identical to a 2-layer candidate from this sweep, so
BEST2-vs-SHIPPED is a controlled head-to-head at 45.780 GiB. See
V2-SWEEP-PLAN.md, "The probe verdict".

WHY A NEW DRIVER. This mirrors research/qwen397b/v2_sweep.py and shares its
TSV schema so both ledgers read with one set of tooling, but three things
differ and each one matters:

  1. THE PROMOTION CHANGES d, NOT JUST K. 397B promotes d4/K256 -> d4/K512
     (K only). Flash promotes d8/K16384 -> d2/K256: the codebook gets
     SMALLER (16384 -> 256 entries) while the subvector dim drops 8 -> 2,
     which is what makes it richer -- 4x more codes per row. The 397B
     driver dies on `dim differs` by design; here that check is inverted
     into a bytes-must-increase check, because bytes are the honest test of
     "richer" when d changes.
  2. THE SCORER IS KL, NOT PERPLEXITY. Flash has a teacher cache on disk
     (flashnext_teacher_topk_prose) and a VALIDATED qwen4_exp streaming
     scorer (stream_score.py:181-183). KL is the ranking column
     (TABLE.md:22,25). --tokens is pinned to 2048: the scorer HARD FAILS if
     token ids differ from the cache (stream_score.py:255-258), so there is
     no cheap-prefix mode to get wrong.
  3. THE INTERPRETER IS PR #1788 mlx-lm, NOT STOCK. qwen4_exp does not
     exist in stock mlx-lm. See --python.

SAFETY. Nothing here loads a model. Splicing is a CPU-only byte transform;
scoring is delegated to vqlab.stream_score, which materializes one
DecoderLayer at a time (stream_score.py:1-8) and is flat in memory. The
preflight guards the WORKING SET this driver actually allocates (one shard
group in memory for the rewrite) and refuses to start if it cannot fit.

USAGE

  ./flash_v2_sweep.py --dry-run                    # validate, load nothing
  ./flash_v2_sweep.py --verify-instrument          # reproduce TABLE.md first
  ./flash_v2_sweep.py --layers 0-47 --max-candidates 20
  ./flash_v2_sweep.py --combo 0,1 --tag REPRO_SHIPPED --keep   # self-test

STATE. --state (default state.json) lists completed candidates and is
consulted on every start; a completed candidate is skipped. Per-candidate
rows are appended to --tsv (default sweep.tsv) in the order they finish.
Ctrl-C safe: re-run to continue.
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

# The FLAT rung. NOT the shipped 2.1bpw artifact -- that one already has
# L0+L1 promoted (the probe's hot-2), so it is a MIXED base and would make
# every single-layer effect conditional on two arbitrary prior promotions.
# It survived the 2026-08-29 "superseded, cleanup is Noah's call" note
# (LEDGER.md:231-233), which is what makes this sweep cheap.
BASE_DEFAULT = "qwen4exp_vq_packed_d8k16384"          # 44.531 GiB, flat d8/K16384

# The shipped artifact -- the probe's hot-2, and the row this sweep is
# arguing with. Used by --verify-instrument only.
SHIPPED_DIR = "TheDrainFlorist--Qwen3.8-Flash-Next-VQ-2.1bpw"   # 45.780 GiB
# TABLE.md:8, the cells this instrument must reproduce.
SHIPPED_REF = {"kl": 390.09, "top1": 0.788, "prose": 5.9033}

TEACHER_DEFAULT = "flashnext_teacher_topk_prose"       # top_k 64, tokens 2049

DONORS = {
    # name -> (artifact dir, d, k, GiB added per promoted layer)
    # Byte costs are DERIVED from the safetensors headers of the two PACKED
    # artifacts, not assumed. Verified 2026-09-03: switch_mlp bytes/layer go
    # 0.6208 -> 1.2451 GiB, uniform across all 48 layers, and
    # 44.531 + 2 x 0.6243 = 45.780 -- the shipped artifact's index
    # total_size to the byte.
    "d2k256": ("qwen4exp_vq_fit_d2k256", 2, 256, 0.6243),
    # The 5.5 rung's fit: a second, richer donor. Full 48-layer coverage
    # too. For the step-size control only -- its byte cost is NOT 0.6243 and
    # the driver re-derives it from the indexes at preflight.
    "d2k1024": ("qwen4exp_vq_fit_d2k1024", 2, 1024, None),
}
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
VQ_SUFFIXES = ("codebook", "codes", "vq_scales")
N_VQ_LAYERS = 48            # vq_modules covers layers 0-47, no gaps (144/3)

# The probe's published picks, for the rank-correlation output. Hot and
# quiet sets from LEDGER.md:285-287 (identical at the 2.0 and 3.1 rungs,
# Pearson r=0.905 across all 48 layers -- self-consistency, which this
# sweep tests against actual measurement).
PROBE_HOT = [0, 1, 31, 32, 33, 35, 36, 37, 38, 39]
PROBE_QUIET = [3, 8, 9, 10, 11, 13, 16, 23, 45, 46]
SHIPPED_LAYERS = [0, 1]     # the probe's hot-2, as actually shipped


def die(msg, code=2):
    print(f"PREFLIGHT FAIL: {msg}", file=sys.stderr)
    sys.exit(code)


def module_name(layer, proj):
    return f"model.layers.{layer}.mlp.switch_mlp.{proj}"


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


def safetensors_header(path):
    """Read a shard's JSON header without materializing any tensor."""
    import struct
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n))


def switch_mlp_bytes_per_layer(root, wm):
    """{layer: bytes} for the switch_mlp VQ tensors only, from headers.

    Header-only: opens each shard that carries a switch_mlp tensor and reads
    its data_offsets. Nothing is loaded, so this is safe to run against a
    100 GiB artifact on a contended box.
    """
    import collections
    import re
    shards = collections.defaultdict(list)
    for k, v in wm.items():
        if ".mlp.switch_mlp." in k:
            shards[v].append(k)
    per = collections.defaultdict(int)
    for sh, keys in shards.items():
        h = safetensors_header(pathlib.Path(root) / sh)
        for k in keys:
            layer = int(re.search(r"layers\.(\d+)\.", k).group(1))
            off = h[k]["data_offsets"]
            per[layer] += off[1] - off[0]
    return dict(per)


def physical_ram_gib():
    try:
        n = subprocess.run(["sysctl", "-n", "hw.memsize"],
                           capture_output=True, text=True, check=True)
        return int(n.stdout.strip()) / GiB
    except Exception:
        return None


# --------------------------------------------------------------------------
# preflight -- refuse to start rather than thrash

def preflight(base, donor, args, layers):
    bp, bidx, bcfg = load_artifact(base)
    dp, didx, dcfg = load_artifact(donor)
    bwm, dwm = bidx["weight_map"], didx["weight_map"]

    # --- the splice contract (research/allocation/METHOD.md:193-199) -------
    if set(bwm) != set(dwm):
        die(f"key sets differ: base {len(bwm)} vs donor {len(dwm)} tensors. "
            "Shard-level hardlinking needs identical key sets; relax to a "
            "per-key rewrite + index rebuild first (METHOD.md:193-199).")
    mismatched = [k for k in bwm if bwm[k] != dwm[k]]
    if mismatched:
        die(f"{len(mismatched)} tensors sit in different shards between base "
            f"and donor (e.g. {mismatched[0]}). Shard-level hardlinking is "
            "invalid here.")

    bvm, dvm = bcfg.get("vq_modules"), dcfg.get("vq_modules")
    if not bvm or not dvm:
        die("one of the artifacts has no vq_modules block in config.json")
    if set(bvm) != set(dvm):
        die("vq_modules coverage differs between base and donor")

    # --- geometry ---------------------------------------------------------
    bk = {v["k"] for v in bvm.values()}
    dk = {v["k"] for v in dvm.values()}
    bd = {v["dim"] for v in bvm.values()}
    dd = {v["dim"] for v in dvm.values()}
    if len(bk) != 1 or len(bd) != 1:
        die(f"the base is NOT FLAT: k={sorted(bk)} dim={sorted(bd)}. This "
            "driver measures single-layer effects against a flat reference; "
            f"a mixed base (e.g. the shipped {SHIPPED_DIR}, which already "
            "carries L0+L1 at d2/K256) makes every effect conditional on "
            "prior promotions. Use --base qwen4exp_vq_packed_d8k16384.")
    if len(dk) != 1 or len(dd) != 1:
        die(f"the donor is not flat: k={sorted(dk)} dim={sorted(dd)}")

    # NOTE: unlike the 397B driver we ALLOW dim to change -- this family's
    # promotion is d8/K16384 -> d2/K256, where the codebook shrinks and the
    # subvector dim shrinks harder. "Richer" is therefore tested in BYTES,
    # which is the honest test when d moves.
    b_tot = bidx.get("metadata", {}).get("total_size")
    d_tot = didx.get("metadata", {}).get("total_size")
    if b_tot is None or d_tot is None:
        die("an index is missing metadata.total_size; byte cost cannot be "
            "derived and must never be assumed")

    # Byte cost is derived from the switch_mlp TENSOR HEADERS, scoped to the
    # tensors this driver actually swaps.
    #
    # It is NOT (d_tot - b_tot) / N_VQ_LAYERS, the way the 397B driver does
    # it. That shortcut is only valid when base and donor differ in expert
    # geometry ALONE. Here the donor is a whole d2/K256 model whose PLE bank
    # is a different geometry too, so the whole-index delta gives 0.9968
    # GiB/layer -- 60% too high, and it would silently corrupt every budget
    # and total-size claim downstream. Scoped, the answer is 0.6243, which
    # reproduces the shipped artifact exactly (see the assertion below).
    b_sw = switch_mlp_bytes_per_layer(bp, bwm)
    d_sw = switch_mlp_bytes_per_layer(dp, dwm)
    if len(set(b_sw.values())) != 1 or len(set(d_sw.values())) != 1:
        die("switch_mlp bytes/layer are not uniform; the per-layer promotion "
            "cost is not a single number and the budget maths below is void")
    per_layer_gib = (d_sw[0] - b_sw[0]) / GiB
    if per_layer_gib <= 0:
        die(f"donor geometry d{sorted(dd)[0]}/K{sorted(dk)[0]} is not richer "
            f"than base d{sorted(bd)[0]}/K{sorted(bk)[0]}: it would ADD "
            f"{per_layer_gib:.4f} GiB/layer. A promotion must cost bytes.")

    # Falsification check: 2 promoted layers must land exactly on the shipped
    # artifact's index total_size. If this drifts, the step is wrong and
    # every size in V2-SWEEP-PLAN.md is wrong with it.
    shipped_idx = pathlib.Path(args.models_root) / SHIPPED_DIR \
        / "model.safetensors.index.json"
    if shipped_idx.exists():
        s_tot = json.load(open(shipped_idx)).get("metadata", {}).get(
            "total_size")
        if s_tot:
            pred = (b_tot + 2 * (d_sw[0] - b_sw[0])) / GiB
            got = s_tot / GiB
            if abs(pred - got) > 0.002:
                die(f"step check FAILED: flat base + 2 promotions predicts "
                    f"{pred:.3f} GiB but the shipped hot-2 artifact is "
                    f"{got:.3f} GiB. The promotion step is wrong.")
            print(f"PREFLIGHT  step check OK: base + 2 promotions = "
                  f"{pred:.3f} GiB == shipped hot-2 {got:.3f} GiB")

    for layer in layers:
        for proj in PROJECTIONS:
            m = module_name(layer, proj)
            if m not in bvm:
                die(f"layer {layer}: {m} is not a VQ module in the base")

    # --- working set: shards touched, and the largest group ---------------
    touched = {}
    for layer in layers:
        s = set()
        for proj in PROJECTIONS:
            for suf in VQ_SUFFIXES:
                s.add(bwm[f"{module_name(layer, proj)}.{suf}"])
        touched[layer] = sorted(s)
    group_gib = {
        l: sum(os.path.getsize(bp / f) for f in sh) / GiB
        for l, sh in touched.items()
    }
    biggest = max(group_gib.values())
    median = sorted(group_gib.values())[len(group_gib) // 2]

    # A rewrite holds the shard group's tensors in memory (base side and
    # donor side) while writing the replacement: budget 2x plus slack.
    need = biggest * 2 + 2.0
    ram = physical_ram_gib()
    if ram is not None:
        bar = ram * args.headroom
        ok = need <= bar
        print(f"PREFLIGHT  splice working set {need:.2f} GiB "
              f"(largest shard group {biggest:.2f} GiB x2 + 2.0)   "
              f"box RAM {ram:.0f} GiB   bar {bar:.1f} GiB "
              f"({args.headroom:.0%})   -> {'OK' if ok else 'FAIL'}")
        if not ok:
            die(f"the splice working set ({need:.2f} GiB) does not fit under "
                f"{bar:.1f} GiB on this box. Run on a bigger box, or sweep "
                "only the layers whose shard groups are small "
                f"(median here is {median:.2f} GiB).")
    else:
        print("PREFLIGHT  WARN: could not read hw.memsize; RAM bar not checked")
    print(f"PREFLIGHT  scorer is streaming (vqlab.stream_score, one "
          "DecoderLayer at a time -- stream_score.py:1-8); a 598 GiB teacher "
          "scores on a 96 GB box, so a 45 GiB candidate is not resident-bound")

    # --- disk -------------------------------------------------------------
    probe_dir = args.workdir if os.path.isdir(args.workdir) else E_DEFAULT
    stv = os.statvfs(probe_dir)
    free = stv.f_bavail * stv.f_frsize / GiB
    concurrent = len(layers) if args.keep else 1
    # only the rewritten shard group costs bytes; the rest is hardlinked
    disk_need = biggest * 1.05 * concurrent
    print(f"PREFLIGHT  disk need {disk_need:.1f} GiB "
          f"({'all candidates kept' if args.keep else 'one at a time'}; "
          "untouched shards are hardlinked)   "
          f"free {free:.1f} GiB   -> {'OK' if disk_need <= free else 'FAIL'}")
    if disk_need > free:
        die(f"need {disk_need:.1f} GiB of scratch and only {free:.1f} GiB is "
            "free. Drop --keep, or free space.")

    # --- instruments must exist before anything runs ----------------------
    if not os.path.exists(args.corpus):
        die(f"corpus not found: {args.corpus}")
    if not os.path.isdir(args.teacher):
        die(f"teacher KL cache not found: {args.teacher}")
    meta_p = os.path.join(args.teacher, "meta.json")
    if not os.path.exists(meta_p):
        die(f"teacher cache has no meta.json: {args.teacher}")
    meta = json.load(open(meta_p))
    # The scorer refuses if token ids differ from the cache
    # (stream_score.py:255-258). Catch it here instead of 20 minutes in.
    want = meta.get("tokens", 0) - 1
    if args.tokens != want:
        die(f"--tokens {args.tokens} but the teacher cache holds "
            f"{meta['tokens']} ids (= {want} scored positions). The scorer "
            "hard-fails on a token-id mismatch; there is no degraded mode.")
    if not os.path.exists(args.python):
        die(f"interpreter not found: {args.python}")
    if "envs/exo" in args.python:
        die(f"--python points at an exo env ({args.python}). Those carry a "
            "grafted mlx-lm and a jaccl mlx fork; a number from them is "
            "OFF-INSTRUMENT. The 2026-09-03 GLM incident "
            "(glm53-flash/LEDGER.md:98-116) is exactly this, and the "
            "divergence is not uniform so it cannot be corrected out.")

    print(f"PREFLIGHT  base   {bp.name}  {b_tot / GiB:.3f} GiB  "
          f"FLAT d{sorted(bd)[0]}/K{sorted(bk)[0]}")
    print(f"PREFLIGHT  donor  {dp.name}  {d_tot / GiB:.3f} GiB  "
          f"d{sorted(dd)[0]}/K{sorted(dk)[0]}")
    print(f"PREFLIGHT  promotion cost {per_layer_gib:.4f} GiB/layer  "
          f"({len(layers)} candidates queued)")
    print(f"PREFLIGHT  shard groups: median {median:.2f} GiB, "
          f"max {biggest:.2f} GiB  "
          f"({sum(1 for v in touched.values() if len(v) > 1)} layers span "
          "2 shards)")
    print(f"PREFLIGHT  interpreter {args.python}")
    print("PREFLIGHT  OK")
    return dict(bp=bp, dp=dp, bwm=bwm, bvm=bvm, dvm=dvm,
                per_layer_gib=per_layer_gib, touched=touched,
                b_tot=b_tot, biggest=biggest, median=median)


# --------------------------------------------------------------------------
# splice

def splice(pf, layers, out_dir, verbose=True):
    """Build an artifact = base with `layers` promoted to the donor geometry.

    Untouched shards are HARDLINKED (no copy, no extra bytes). Touched
    shards are rewritten with the donor's codebook/codes/vq_scales for the
    promoted modules and the base's tensors for everything else.
    """
    import mlx.core as mx
    # CPU-only, for the same reason splice_ple.py does it: this is a pure
    # byte transform over shards on an external volume, and a GPU command
    # buffer wrapped around a slow read trips the Metal watchdog. The
    # layer-leverage probe hit exactly that on the M4 over SMB
    # (LEDGER.md:293-296).
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

    # 1. every non-shard file comes across as-is. NOTE this deliberately
    #    copies no directories and no sidecars: mtp-head-q6.safetensors does
    #    not match mlx-lm's model*.safetensors glob (LEDGER.md:490-493) and
    #    is not part of a quality measurement anyway (the trunk verifies
    #    every drafted token, LEDGER.md:485-487).
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

    # 3. config.json -- promoted modules take the donor's vq_modules entry.
    #    This matters beyond bookkeeping: the base entry carries
    #    pack_bits 14 and the donor's does not (u8 codes need no unpack), so
    #    a stale entry would send the runtime down the wrong load path.
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

def score(args, model_dir):
    """One streamed pass -> {ppl, mean_kl_millinats, top1_agreement, ...}."""
    cmd = [args.python, "-m", "vqlab.stream_score",
           "--model", str(model_dir),
           "--corpus", args.corpus,
           "--tokens", str(args.tokens),
           "--kl-cache", args.teacher]
    t0 = time.time()
    env = dict(os.environ)
    # run from the repo root so `-m vqlab.stream_score` resolves
    p = subprocess.run(cmd, capture_output=True, text=True,
                       cwd=args.repo_root, env=env)
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
    if "mean_kl_millinats" not in rec:
        raise RuntimeError(
            "scorer returned no KL -- the --kl-cache did not engage. KL is "
            "the ranking column (TABLE.md:22,25); a ppl-only row is not a "
            "sweep row.")
    if rec.get("unvalidated"):
        raise RuntimeError(
            "the scorer stamped this record UNVALIDATED (rule 5). Such a "
            "number must never enter a ladder.")
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


# First 12 columns are byte-compatible with research/qwen397b/v2_sweep.py so
# both ledgers read with one set of tooling; the rest are this family's
# (KL is the ranking column here, and every number names its interpreter).
TSV_HEADER = ["candidate", "layers", "donor", "gib", "d_gib",
              "prose_ppl", "d_prose", "code_ppl", "d_code",
              "tokens", "wall_s", "utc",
              "kl_mnats", "d_kl", "top1", "captured_mass",
              "interpreter", "contended"]


def verify_instrument(args, pf):
    """Re-score the SHIPPED artifact and refuse to continue on a mismatch.

    Closes the open question of WHICH interpreter cut TABLE.md before 48
    candidates are spent on the wrong one. See V2-SWEEP-PLAN.md Q3.
    """
    shipped = os.path.join(args.models_root, SHIPPED_DIR)
    if not os.path.isdir(shipped):
        die(f"shipped artifact not found: {shipped}")
    print(f"\n=== verifying the instrument against TABLE.md:8 ===")
    print(f"  artifact    {SHIPPED_DIR}")
    print(f"  interpreter {args.python}")
    print(f"  expecting   KL {SHIPPED_REF['kl']} / top-1 "
          f"{SHIPPED_REF['top1']:.1%} / prose {SHIPPED_REF['prose']}",
          flush=True)
    rec = score(args, shipped)
    got_kl = rec["mean_kl_millinats"]
    got_t1 = rec["top1_agreement"]
    got_p = rec["ppl"]
    print(f"  measured    KL {got_kl} / top-1 {got_t1:.1%} / prose {got_p}"
          f"   ({rec['wall_s']}s)")
    # TABLE.md prints KL to 2dp, top-1 to 0.1%, prose to 4dp. Compare at the
    # printed precision -- the GLM incident's whole point is that a
    # different MLX build shifts digits that DO print.
    bad = []
    if round(got_kl, 2) != SHIPPED_REF["kl"]:
        bad.append(f"KL {round(got_kl, 2)} != {SHIPPED_REF['kl']}")
    if round(got_t1, 3) != SHIPPED_REF["top1"]:
        bad.append(f"top-1 {round(got_t1, 3)} != {SHIPPED_REF['top1']}")
    if round(got_p, 4) != SHIPPED_REF["prose"]:
        bad.append(f"prose {round(got_p, 4)} != {SHIPPED_REF['prose']}")
    if bad:
        die("INSTRUMENT MISMATCH: " + "; ".join(bad) + ".\n"
            f"  {args.python} did NOT cut TABLE.md. Every number from it "
            "would be off-instrument and could not be compared against the "
            "shipped rung's KL 390.09 -- which is the whole point of this "
            "sweep. Find the interpreter that reproduces these cells "
            "(candidates: ~/.venvs/qwen4exp, "
            "'/Volumes/Thunderbay SSD/venvs/qwen4exp') and re-run. Do NOT "
            "proceed with an offset; the 2026-09-03 divergence was not "
            "uniform across corpora (glm53-flash/LEDGER.md:158-164).")
    print("  INSTRUMENT OK -- reproduces TABLE.md:8 to the printed decimal.")
    return rec


def report_probe_verdict(st):
    """Rank the measured effects and score the probe against them."""
    rows = [(k, v) for k, v in st["done"].items()
            if v.get("layers") and "," not in str(v["layers"])]
    if len(rows) < 4:
        return
    eff = {int(v["layers"]): float(v["d_kl"]) for _, v in rows if v.get("d_kl")}
    if not eff:
        return
    ranked = sorted(eff, key=lambda l: -eff[l])       # biggest KL drop first
    print(f"\n=== measured ranking so far ({len(ranked)}/{N_VQ_LAYERS} layers) ===")
    print("  best 8 by measured effect:", ranked[:8])
    print("  worst 8 by measured effect:", ranked[-8:])
    hot = [l for l in PROBE_HOT if l in eff]
    quiet = [l for l in PROBE_QUIET if l in eff]
    if hot:
        pos = [ranked.index(l) + 1 for l in hot]
        print(f"  probe's hot set {hot}")
        print(f"    -> measured ranks {pos}  (1 = biggest measured effect)")
    if quiet:
        pos = [ranked.index(l) + 1 for l in quiet]
        print(f"  probe's quiet set {quiet}")
        print(f"    -> measured ranks {pos}")
    ship = [l for l in SHIPPED_LAYERS if l in eff]
    if ship:
        print(f"  SHIPPED hot-2 {ship} -> measured ranks "
              f"{[ranked.index(l) + 1 for l in ship]}")
    harmful = [l for l in ranked if eff[l] < 0]
    if harmful:
        print(f"  layers whose promotion made KL WORSE: {harmful}")
        overlap = sorted(set(harmful) & set(PROBE_HOT))
        if overlap:
            print(f"    ...of which the probe called HOT: {overlap}  "
                  "<-- the GLM failure mode, reproduced on Flash")
    if len(ranked) == N_VQ_LAYERS:
        print(f"\n  VERDICT ROWS TO BUILD (byte-matched at "
              f"{44.531 + 2 * 0.6243:.3f} GiB):")
        print(f"    --combo {ranked[0]},{ranked[1]} --tag BEST2")
        print(f"    --combo {ranked[-2]},{ranked[-1]} --tag CTRL2")
        print("    SHIPPED is already scored: KL 390.09 (TABLE.md:8)")


def main():
    ap = argparse.ArgumentParser(
        description="Flash-Next v2 mixed-codebook single-layer sweep")
    ap.add_argument("--models-root", default=E_DEFAULT)
    ap.add_argument("--base", default=BASE_DEFAULT,
                    help="base artifact dir under --models-root. MUST be "
                         "flat; the shipped 2.1bpw is NOT (it carries the "
                         "probe's hot-2 already)")
    ap.add_argument("--donor", default="d2k256", choices=sorted(DONORS),
                    help="richer geometry to promote INTO (default d2k256, "
                         "the shipped mix's donor: +0.6243 GiB/layer)")
    ap.add_argument("--layers", default="0-47",
                    help="single-layer candidates to sweep, e.g. 0-47 or 3,7")
    ap.add_argument("--combo", default=None,
                    help="build ONE candidate promoting all these layers "
                         "(stage 3); implies a single run")
    ap.add_argument("--tag", default=None, help="name for a --combo candidate")
    ap.add_argument("--workdir", default=None,
                    help="where candidates are built (default <models-root>"
                         "/v2sweepflash)")
    ap.add_argument("--state", default=None)
    ap.add_argument("--tsv", default=None)
    ap.add_argument("--tokens", type=int, default=2048,
                    help="scored positions. PINNED to the teacher cache: the "
                         "scorer hard-fails on a token-id mismatch, so this "
                         "is not a speed dial")
    ap.add_argument("--keep", action="store_true",
                    help="keep every candidate on disk instead of deleting "
                         "after scoring")
    ap.add_argument("--budget-gib", type=float, default=1.25,
                    help="promotion budget over the flat base; the driver "
                         "WARNS when a combo exceeds it. Default 1.25 = "
                         "byte-matched to the shipped rung (2 layers)")
    ap.add_argument("--headroom", type=float, default=0.90,
                    help="fraction of RAM the splice working set may take")
    ap.add_argument("--max-candidates", type=int, default=None,
                    help="stop after this many NEW candidates this run "
                         "(use to fit a night)")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter for the scorer. MUST be a qwen4_exp "
                         "runtime (mlx-lm 0.32.0 from the unmerged PR "
                         "ml-explore/mlx-lm#1788) -- stock mlx-lm has no "
                         "qwen4_exp. On M3 and M4 that is "
                         "~/.venvs/qwen4exp/bin/python (docs/MTP.md:377). "
                         "NEVER an exo env: grafted mlx-lm + jaccl mlx fork, "
                         "so any number from them is off-instrument "
                         "(glm53-flash/LEDGER.md:98-116)")
    ap.add_argument("--corpus", default=None,
                    help="default: the repo's referee_corpus.txt, which is "
                         "the corpus the teacher cache was cut against")
    ap.add_argument("--teacher", default=None,
                    help="teacher top-64 cache dir "
                         f"(default <models-root>/{TEACHER_DEFAULT})")
    ap.add_argument("--contended", action="store_true",
                    help="mark rows as produced while another job held the "
                         "boxes. KL is deterministic and unaffected; "
                         "wall_s is NOT trustworthy (docs/MTP.md:405-410)")
    ap.add_argument("--verify-instrument", action="store_true",
                    help="re-score the shipped 2.1bpw and refuse to continue "
                         "unless it reproduces TABLE.md:8, then exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate paths, geometry and budget; load nothing")
    a = ap.parse_args()

    here = pathlib.Path(__file__).resolve().parent
    # research/quantlab/research/flash-next -> repo root
    a.repo_root = str(here.parents[3])
    a.corpus = a.corpus or str(
        pathlib.Path(a.repo_root) / "src" / "vqlab" / "referee"
        / "referee_corpus.txt")
    a.teacher = a.teacher or os.path.join(a.models_root, TEACHER_DEFAULT)
    a.workdir = a.workdir or os.path.join(a.models_root, "v2sweepflash")
    a.state = a.state or str(here / "state.json")
    a.tsv = a.tsv or str(here / "sweep.tsv")

    base = os.path.join(a.models_root, a.base)
    donor_name, dd, dk, _ = DONORS[a.donor]
    donor = os.path.join(a.models_root, donor_name)
    for p in (base, donor):
        if not os.path.isdir(p):
            die(f"artifact not found: {p}")

    if a.combo:
        layers = parse_layers(a.combo)
        tag = a.tag or ("mix" + "_".join(map(str, layers)))
        combos = [(tag, layers)]
        sweep_layers = layers
    else:
        sweep_layers = parse_layers(a.layers)
        combos = [(f"L{l:02d}", [l]) for l in sweep_layers]

    pf = preflight(base, donor, a, sweep_layers)

    # budget warning, never a hard stop -- stages 4 and 5 exceed it by design
    for tag, layers in combos:
        cost = len(layers) * pf["per_layer_gib"]
        if cost > a.budget_gib + 1e-9:
            print(f"PREFLIGHT  WARN: {tag} promotes {len(layers)} layers = "
                  f"+{cost:.3f} GiB, over the {a.budget_gib:.2f} GiB budget "
                  f"(total {pf['b_tot'] / GiB + cost:.3f} GiB). "
                  "See V2-SWEEP-PLAN.md Q1 for the tiers.")

    if a.dry_run:
        print("\nDRY RUN -- nothing was loaded, nothing was written.")
        print(f"  base        {a.base}  ({pf['b_tot'] / GiB:.3f} GiB, flat)")
        print(f"  donor       {donor_name}  (d{dd}/K{dk})")
        print(f"  workdir     {a.workdir}")
        print(f"  state       {a.state}")
        print(f"  tsv         {a.tsv}")
        print(f"  corpus      {a.corpus}")
        print(f"  teacher     {a.teacher}")
        print(f"  interpreter {a.python}")
        print(f"  candidates  {len(combos)}  "
              f"(first {[t for t, _ in combos[:5]]})")
        print(f"  each promotes to d{dd}/K{dk} at "
              f"+{pf['per_layer_gib']:.4f} GiB/layer")
        print(f"  per candidate: rewrite {pf['median']:.2f} GiB of shards "
              f"(median; max {pf['biggest']:.2f}), hardlink the rest")
        st = load_state(a.state)
        print(f"  already done: {len(st['done'])}")
        print("\n  NEXT: --verify-instrument, before spending a night.")
        return

    if a.verify_instrument:
        verify_instrument(a, pf)
        return

    os.makedirs(a.workdir, exist_ok=True)
    st = load_state(a.state)
    st["base"] = a.base
    st.setdefault("donor", a.donor)
    st.setdefault("interpreter", a.python)
    if st["donor"] != a.donor:
        die(f"state file was built against donor {st['donor']}, not "
            f"{a.donor}. Layer effects are base- and donor-specific; "
            "use a separate --state.")
    if st["interpreter"] != a.python:
        die(f"state file was built on interpreter {st['interpreter']}, not "
            f"{a.python}. Mixing interpreters inside one sweep is the "
            "2026-09-03 GLM incident (glm53-flash/LEDGER.md:98-116); the "
            "divergence is not uniform, so the rows would not be "
            "comparable. Use a separate --state.")

    # baseline score, once
    if "BASE" not in st["done"]:
        print("\n=== scoring the flat base (reference row) ===", flush=True)
        rec = score(a, base)
        row = {"candidate": "BASE", "layers": "", "donor": a.donor,
               "gib": round(pf["b_tot"] / GiB, 3), "d_gib": 0.0,
               "prose_ppl": rec["ppl"], "d_prose": 0.0,
               "kl_mnats": rec["mean_kl_millinats"], "d_kl": 0.0,
               "top1": rec["top1_agreement"],
               "captured_mass": rec["captured_mass"],
               "tokens": a.tokens, "wall_s": rec["wall_s"],
               "interpreter": a.python,
               "contended": int(bool(a.contended)),
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        st["done"]["BASE"] = row
        save_state(a.state, st)
        append_tsv(a.tsv, row, TSV_HEADER)
        print(f"  BASE KL {row['kl_mnats']} mnats  prose {row['prose_ppl']}  "
              f"({row['wall_s']}s)", flush=True)

    base_kl = st["done"]["BASE"]["kl_mnats"]
    base_prose = st["done"]["BASE"]["prose_ppl"]

    n_new = 0
    for tag, layers in combos:
        if tag in st["done"]:
            continue
        if a.max_candidates is not None and n_new >= a.max_candidates:
            print(f"\nstopping: --max-candidates {a.max_candidates} reached")
            break
        print(f"\n=== {tag}  (layers {layers}) ===", flush=True)
        t0 = time.time()
        cand = os.path.join(a.workdir, f"flash-v2-{tag}")
        try:
            size = splice(pf, layers, cand)
            rec = score(a, cand)
            row = {"candidate": tag, "layers": ",".join(map(str, layers)),
                   "donor": a.donor, "gib": round(size, 3),
                   "d_gib": round(size - pf["b_tot"] / GiB, 4),
                   "prose_ppl": rec["ppl"],
                   "d_prose": round(base_prose - rec["ppl"], 6),
                   "kl_mnats": rec["mean_kl_millinats"],
                   # positive d_kl = promotion HELPED (KL fell)
                   "d_kl": round(base_kl - rec["mean_kl_millinats"], 4),
                   "top1": rec["top1_agreement"],
                   "captured_mass": rec["captured_mass"],
                   "tokens": a.tokens,
                   "interpreter": a.python,
                   "contended": int(bool(a.contended)),
                   "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
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
        print(f"  {tag}: KL {row['kl_mnats']}  d_kl {row['d_kl']:+.2f} mnats  "
              f"prose {row['prose_ppl']}  {rate:.0f}s   "
              f"[{done}/{len(combos)} done, ~"
              f"{(len(combos) - done) * rate / 3600:.1f}h left at this rate]",
              flush=True)

    report_probe_verdict(st)
    print(f"\nwrote {a.tsv} and {a.state}")


if __name__ == "__main__":
    main()
