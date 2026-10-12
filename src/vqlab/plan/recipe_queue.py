"""vqlab recipe — compile a build recipe into a `vqlab queue` file.

    vqlab recipe compile recipe.json [--out queue.json]

A recipe says WHAT to build; this writes HOW, as the steps a queue runs
(queue/runner.py): pinned code, the GPU lease, preflight, loud failure.
Every step is a vqlab command with the flags the lab's queues of record ran
it with; a chain is offered only for families whose pieces have each run:

  MoE (qwen3_5):  stream-convert --struct -> fit-moe --pack [--tail-from/
                  --tail-geom: a mixed build] -> [graft -> vision-smoke] -> pin
                  -> check-bundle --strict -> check-release -> [kl-ladder]
  dense (qwen3_8_dense):  [stream-convert --bits: the base, unless an
                  incumbent is given] -> fit-dense -> build-dense --dry-run ->
                  build-dense -> pack-dense -> pin -> checks -> [kl-ladder]
  affine (either):  stream-convert --bits -> pin -> checks -> [kl-ladder]

A compiled chain is proven by its PREFLIGHT, which runs every step small
and real (the fit on one layer, plus the tail's first layer for a mixed
build); start the full run only from a preflight that passed. Any other
family is refused by name. The MTP head is never compiled: its wiring flags
come from `vqlab mtp-probe35`, and a wrong guess gives zero acceptance with
no error. kl-ladder is added only when the teacher's caches are registered
(families/<f>/teachers/<t>/teacher_caches.json); it scores absolute KL, not
the paired release gate. A fit's --relerr-abort comes from measured healthy
ranges only (FINDINGS I.8); an unmeasured geometry is refused.

Recipe (JSON):

    {"name": "q35-d4k256", "family": "qwen3_5", "teacher": "Qwen--Qwen3.6-35B-A3B-bf16",
     "mode": "vq", "geo": "d4-K256",
     "tail": {"from": 30, "geo": "d4-K512"},      optional: a mixed build
     "bits": 4, "group": 64,                       affine (and the dense base)
     "protect_bits": 8,                            MoE: attention/embeddings/shared (struct base)
     "base": "<models>/org--name-4bit",            dense: the quantized incumbent
     "vision": true, "score": true}

The teacher is a profiled teacher (`vqlab family-profile`); its profile
gives the source path and the layers. Every path in the queue is written as
<models>/..., <scratch>/... or hf://org/name@rev, so a queue file names no
machine; each step expands them where it runs.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

from vqlab import config
from vqlab.family import registry

MOE = {"qwen3_5"}
DENSE = {"qwen3_8_dense"}
GEO = re.compile(r"^d(\d+)-?[kK](\d+)$")
NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,79}$")


class RecipeError(ValueError):
    """A recipe that cannot be compiled; the message says why."""


def profile(family: str, teacher: str) -> dict:
    """The teacher's family profile (checkout families/, then the user's)."""
    for d in registry._family_dirs():
        p = pathlib.Path(d) / family / "teachers" / teacher / "profile.json"
        if p.is_file():
            return json.loads(p.read_text())
    raise RecipeError(f"{family}/{teacher} has no profile: run `vqlab family-profile --teacher <dir>` first")


def source_path(path: str) -> str:
    """The teacher path as recorded (<models>/..., hf://org/name@rev), which
    each step expands on the machine that runs it; refused here when THIS
    machine cannot expand it."""
    local = config.expand(path)
    if local.startswith("hf://") or not pathlib.Path(local).is_dir():
        raise RecipeError(f"the teacher {path} is not on this machine")
    return path


def geometry(geo: str) -> tuple[int, int]:
    m = GEO.match(str(geo or ""))
    if not m:
        raise RecipeError(f"geometry {geo!r} is not dD-KK (e.g. d4-K256)")
    return int(m.group(1)), int(m.group(2))


def relerr_abort(d: int, k: int) -> float:
    """fit-moe's abort threshold, only where a healthy relerr was measured.
    Healthy relerr scales with K and d (FINDINGS I.8: K2048 ~0.19, K256
    ~0.31, K128 ~0.46; d8 ~0.32-0.35; d2 K2048 ~0.03), so one threshold
    aborts healthy fits at another geometry; an unmeasured one has none."""
    if d == 8 and k >= 4096:
        return 0.45
    if d == 4 and k >= 512:
        return 0.35
    if d == 4 and k == 256:
        return 0.45
    if d == 4 and k == 128:
        return 0.6
    if d == 2 and k >= 256:
        return 0.35
    raise RecipeError(f"d{d}-K{k} has no measured healthy relerr to set the fit's abort "
                      "threshold; fit it by hand first (`vqlab fit-moe --relerr-abort`)")


def _legal(prof: dict, d: int, k: int) -> None:
    """Refuse a geometry the profile does not mark legal (exact packing and
    the fused kernel's 32 KiB codebook ceiling, FINDINGS IV, are folded in)."""
    for proj, rows in prof.get("geometries", {}).items():
        row = next((r for r in rows if r["d"] == d and r["K"] == k), None)
        if row is None or not row.get("legal") or not row.get("fused_kernel_ok", True):
            raise RecipeError(f"d{d}-K{k} is not a legal geometry for {proj} on this teacher")


def _stack_gib(prof: dict, d: int, k: int) -> float:
    """The expert/MLP stack at (d, K), summed over projections: the profile's
    exact bytes for that geometry."""
    return sum(next(r for r in rows if r["d"] == d and r["K"] == k).get("stack_gib", 0.0)
               for rows in prof["geometries"].values())


def _caches(teacher: str) -> list[str]:
    """`corpus=path` for the teacher's registered KL caches, one per corpus."""
    got = {}
    for c in registry.teacher_caches(teacher):
        if c.get("teacher") == teacher and c.get("corpus") and c.get("path"):
            got.setdefault(c["corpus"], config.portable(c["path"]))
    return [f"{k}={v}" for k, v in sorted(got.items())]


def _step(name, cmd, args, preflight, **kw) -> dict:
    if "args" in preflight:
        preflight = {**preflight, "args": [str(a) for a in preflight["args"]]}
    return {"name": name, "cmd": cmd, "args": [str(a) for a in args], "preflight": preflight, **kw}


def _graft_flags(src: str) -> list[str]:
    """`graft` flags for this teacher's tower layout. An HF-layout teacher
    stores it as model.visual.*, channels-first; the mlx artifact wants
    vision_tower.* with the patch-embed conv channels-last (graft --help:
    measured on the Qwen3.5-35B pair, 332/333 tensors identical, the conv
    permuted)."""
    idx = pathlib.Path(config.expand(src)) / "model.safetensors.index.json"
    try:
        keys = json.loads(idx.read_text())["weight_map"]
    except (OSError, ValueError, KeyError) as e:
        raise RecipeError(f"cannot read the teacher's tensor index ({idx}): {e}")
    if any(k.startswith("model.visual.") for k in keys):
        return ["--dest-prefix", "vision_tower", "--permute-conv5"]
    return []


def _layers(prof: dict) -> tuple[int, int]:
    vq = prof["arch"].get("vq_layers") or []
    if not vq:
        raise RecipeError("the profile lists no VQ layers")
    return min(vq), max(vq)


def compile_recipe(r: dict) -> tuple[dict, list[str]]:
    """-> (queue file dict, warnings). Raises RecipeError."""
    name, family, teacher = r.get("name", ""), r.get("family", ""), r.get("teacher", "")
    if not NAME.match(name):
        raise RecipeError("name: lowercase letters, digits, '.', '_' or '-', at most 80")
    mode = r.get("mode", "vq")
    if mode not in ("vq", "affine"):
        raise RecipeError(f"mode {mode!r}: vq or affine")
    if family not in MOE | DENSE:
        raise RecipeError(f"no build of family {family!r} has run through a queue yet; "
                          f"compiled families: {', '.join(sorted(MOE | DENSE))}")
    prof = profile(family, teacher)
    src = source_path(prof["identity"]["path"])
    kind = prof["arch"].get("kind")
    if (family in MOE) != (kind == "moe"):
        raise RecipeError(f"{teacher} is {kind}, but {family} builds are compiled as "
                          f"{'moe' if family in MOE else 'dense'}")
    by = prof["bytes"]["by_class_gib"]
    work = f"<scratch>/{name}"
    out = f"{work}/{name}"
    bits, group = int(r.get("bits", 4)), int(r.get("group", 64))
    steps, warns = [], []

    vis = by.get("vision", 0.0) if r.get("vision") else 0.0
    if mode == "affine":
        art_gib = (by["carry"] + by["embed"] + by["vq_target"]) * (bits + 32 / group) / 16 + vis
        scratch_gib = art_gib
        steps.append(_step("convert", "stream-convert",
                           ["--src", src, "--out", out, "--bits", bits, "--family", family],
                           {"full": "stream-convert has no partial mode",
                            "writes_gib": round(by["carry"] + by["embed"] + by["vq_target"] * bits / 16, 1)},
                           retries=1, timeout_s=6 * 3600, expect={"files": [f"{out}/config.json"]}))
    elif family in MOE:
        d, k = geometry(r.get("geo"))
        _legal(prof, d, k)
        lo, hi = _layers(prof)
        struct = f"{work}/struct"
        pb = int(r.get("protect_bits", 8))
        if not 2 <= pb <= 8:
            raise RecipeError("protect_bits (attention, embeddings, shared modules) must be 2-8")
        skel_gib = (by["carry"] + by["embed"]) * (pb + 0.5) / 16      # affine, group 64
        struct_gib = round(skel_gib + by["vq_target"] * 2.5 / 16, 1)   # experts: 2-bit markers
        steps.append(_step("struct-base", "stream-convert",
                           ["--src", src, "--out", struct, "--struct", "--protect-bits", pb,
                            "--family", family],
                           {"full": "the fit needs the whole struct base", "writes_gib": struct_gib},
                           retries=1, timeout_s=6 * 3600, expect={"files": [f"{struct}/config.json"]}))
        args = ["--base", struct, "--src", src, "--out", out, "--vq-layers", f"{lo}-{hi}",
                "--dim", d, "--k", k, "--family", family, "--seed", 1234, "--pack"]
        abort = relerr_abort(d, k)
        tail = r.get("tail")
        if tail:
            td, tk = geometry(tail.get("geo"))
            _legal(prof, td, tk)
            frm = int(tail.get("from", -1))
            if not lo < frm <= hi:
                raise RecipeError(f"tail.from {frm} must be inside layers {lo + 1}-{hi}")
            args += ["--tail-from", frm, "--tail-geom", f"d{td}k{tk}"]
            if relerr_abort(td, tk) != abort:
                warns.append(f"fit-moe takes one --relerr-abort: the looser of the two geometries' "
                             f"({max(abort, relerr_abort(td, tk))}) applies to every layer")
            abort = max(abort, relerr_abort(td, tk))
        args += ["--relerr-abort", abort]
        n = len(prof["arch"]["vq_layers"])
        ntail = (hi - tail["from"] + 1) if tail else 0
        stack = _stack_gib(prof, d, k) * (n - ntail) / n
        if tail:
            stack += _stack_gib(prof, *geometry(tail["geo"])) * ntail / n
        art_gib = skel_gib + stack + vis
        scratch_gib = struct_gib + art_gib
        one = f"{lo},{tail['from']}" if tail else f"{lo}"
        i = args.index("--vq-layers")
        steps.append(_step("fit", "fit-moe", args,
                           {"args": [*args[:i], "--vq-layers", one, *args[i + 2:]],
                            # fit-moe rewrites every base shard, whatever --vq-layers says
                            "writes_gib": struct_gib},
                           retries=3, timeout_s=48 * 3600, expect={"files": [f"{out}/config.json"]}))
    else:
        d, k = geometry(r.get("geo"))
        _legal(prof, d, k)
        lo, hi = _layers(prof)
        fit, build = f"{work}/fit", f"{work}/build"
        base_gib = (by["carry"] + by["embed"]) * (bits + 32 / group) / 16
        art_gib = base_gib + _stack_gib(prof, d, k) + vis
        scratch_gib = 2 * art_gib + (0 if r.get("base") else base_gib + by["vq_target"] * (bits + 32 / group) / 16)
        if r.get("base"):            # the quantized incumbent, as the shipped dense build used
            base = source_path(r["base"])
        else:
            base = f"{work}/base-q{bits}"
            steps.append(_step("base", "stream-convert",
                               ["--src", src, "--out", base, "--bits", bits, "--family", family],
                               {"full": "the build splices into the whole base",
                                "writes_gib": round(by["carry"] + by["embed"] + by["vq_target"] * bits / 16, 1)},
                               retries=1, timeout_s=6 * 3600, expect={"files": [f"{base}/config.json"]}))
            warns.append("no incumbent base given: the base is made by stream-convert, and the "
                         "build checks every fitted module lands on it (build-dense --dry-run)")
        fargs = ["--src", src, "--out", fit, "--family", family, "--layers", f"{lo}-{hi}",
                 "--dim", d, "--k", k, "--group", group, "--seed", 1234]
        steps.append(_step("fit", "fit-dense", fargs,
                           {"args": [*fargs[:fargs.index("--layers")], "--layers", f"{lo}-{lo}",
                                     *fargs[fargs.index("--layers") + 2:]]},
                           retries=3, timeout_s=24 * 3600))
        steps.append(_step("remap-check", "build-dense",
                           ["--family", family, "--base", base, "--mlp", fit, "--out", build, "--dry-run"],
                           {"full": "validates names only, writes nothing", "writes_gib": 0},
                           retries=0, timeout_s=1800, gpu=False))
        steps.append(_step("build", "build-dense",
                           ["--family", family, "--base", base, "--mlp", fit, "--out", build],
                           {"full": "assembles what the fit wrote", "writes_gib": round(sum(by.values()), 1)},
                           retries=1, timeout_s=3 * 3600))
        steps.append(_step("pack", "pack-dense", ["--src", build, "--out", out],
                           {"full": "packs what the build wrote", "writes_gib": round(sum(by.values()), 1)},
                           retries=1, timeout_s=3 * 3600, expect={"files": [f"{out}/config.json"]}))

    if r.get("vision"):
        if by.get("vision"):
            steps.append(_step("vision", "graft", ["--artifact", out, "--src", src, *_graft_flags(src)],
                               {"full": "grafts the tower in place", "writes_gib": round(by["vision"] + 0.5, 1)},
                               retries=0, timeout_s=3600))
            steps.append(_step("vision-smoke", "vision-smoke", [out, "--max-tokens", 8],
                               {"full": "reads the artifact, writes nothing", "writes_gib": 0},
                               retries=0, timeout_s=3600))
        else:
            warns.append("this teacher has no vision tower; nothing to graft")
    if mode == "vq" and r.get("mtp"):
        warns.append("the MTP head is not compiled: choose its wiring with `vqlab mtp-probe35`, "
                     "then `vqlab mtp-pack` by hand")
    pin = f"{work}/pin"
    steps.append(_step("smoke", "pin", [out, "--out", pin, "--max-tokens", 8],
                       {"full": "a pin is symlinks plus an 8-token smoke", "writes_gib": 0},
                       retries=0, timeout_s=3600))
    steps.append(_step("runtime-check", "check-bundle", ["--artifact", out, "--strict"],
                       {"full": "a read-only check", "writes_gib": 0}, retries=0, timeout_s=600))
    steps.append(_step("release-check", "check-release",
                       ["--artifact", out, "--tokenizer-ref", f"{src}/tokenizer.json"],
                       {"full": "reads the artifact; its smokes are short", "writes_gib": 0},
                       retries=0, timeout_s=2 * 3600))
    if r.get("score"):
        caches = _caches(teacher)
        if caches:
            kl = [a for c in caches for a in ("--cache", c)] + [
                "--rung", f"{name}={pin}", "--per-pos-dir", f"{work}/kl-per-pos",
                "--out", f"{work}/kl.json", "--lazy-over-gb", 16]
            warns.append("the score is this build's absolute KL against the teacher; the release "
                         "gate is a PAIRED kl-ladder against the incumbent rung (FINDINGS: KL is the gate)")
            steps.append(_step("score", "kl-ladder", kl, {"append": ["--preflight"]},
                               retries=1, timeout_s=12 * 3600,
                               expect={"stdout_regex": "KL", "files": [f"{work}/kl.json"]}))
        else:
            warns.append(f"no KL caches are registered for {teacher}: not scored "
                         "(build them with `vqlab stream-score`, then register them)")
    return {"name": name, "commit": "HEAD", "recipe": r,
            # estimates for the confirmation, not measurements: the artifact
            # from the profile's exact per-geometry bytes, scratch = every
            # intermediate the chain leaves (struct base, unpacked build)
            "estimate": {"artifact_gib": round(art_gib, 2), "scratch_gib": round(scratch_gib, 1)},
            "steps": steps}, warns


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vqlab recipe", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compile", help="recipe.json -> a queue file")
    c.add_argument("recipe")
    c.add_argument("--out", help="queue file to write (default: print it)")
    a = ap.parse_args(argv)
    try:
        q, warns = compile_recipe(json.loads(pathlib.Path(a.recipe).read_text()))
    except RecipeError as e:
        raise SystemExit(f"recipe refused: {e}")
    for w in warns:
        print(f"note: {w}", file=sys.stderr)
    text = json.dumps(q, indent=1)
    if a.out:
        pathlib.Path(a.out).write_text(text + "\n")
        print(f"wrote {a.out}; run it with `vqlab queue run {a.out} --preflight`")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
