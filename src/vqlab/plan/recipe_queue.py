"""vqlab recipe — compile a build recipe into a `vqlab queue` file.

    vqlab recipe compile recipe.json [--out queue.json]

A recipe says WHAT to build; this writes HOW, as the steps a queue runs
(queue/runner.py): pinned code, the GPU lease, preflight, loud failure.
Every step is a vqlab command with the flags the lab's queues of record ran
it with; a chain is offered only for families whose pieces have each run:

  MoE (qwen3_5):  stream-convert --struct -> fit-moe --pack [--tail-from/
                  --tail-geom: a mixed build]
  dense (qwen3_8_dense):  [stream-convert --bits: the base, unless an
                  incumbent is given] -> fit-dense -> build-dense --dry-run ->
                  build-dense -> pack-dense
  affine (either):  stream-convert --bits

and every chain, whatever its mode, ends the same way: [graft ->
vision-smoke] -> pin -> check-bundle --strict -> check-release -> [kl-ladder].

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
import dataclasses
import json
import pathlib
import re
import sys

from vqlab import config
from vqlab.family import registry
from vqlab.util.errors import UserError
from vqlab.util.geo import geo_key

MOE = {"qwen3_5"}
DENSE = {"qwen3_8_dense"}
GEO = re.compile(r"^d(\d+)-?[kK](\d+)$")
NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,79}$")


class RecipeError(UserError):
    """A recipe that cannot be compiled; the message says why."""


@dataclasses.dataclass(frozen=True)
class Geometry:
    """One (d, K) the profile marks legal: the expert/MLP stack's exact GiB at
    it, summed over projections, and its bits per weight."""
    d: int
    K: int
    gib: float
    bpw: float | None


def load_profile(family: str, teacher: str) -> dict:
    """The teacher's family profile (registry.teacher_profile: the user's
    over the checkout's)."""
    prof = registry.teacher_profile(family, teacher)
    if prof is None:
        raise RecipeError(f"{family}/{teacher} has no profile (or none readable): run "
                          "`vqlab family-profile --teacher <dir>` first")
    return prof


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
    raise RecipeError(f"{geo_key(d, k)} has no measured healthy relerr to set the fit's abort "
                      "threshold; fit it by hand first (`vqlab fit-moe --relerr-abort`)")


def has_relerr(d: int, k: int) -> bool:
    """Whether a fit at (d, K) can be compiled (relerr_abort has a threshold)."""
    try:
        relerr_abort(d, k)
    except RecipeError:
        return False
    return True


def legal_geometries(prof: dict) -> list[Geometry]:
    """Every (d, K) the profile marks legal on ALL projections (exact packing
    and the fused kernel's 32 KiB codebook ceiling, FINDINGS IV, are folded
    into its table), sorted."""
    tables = prof.get("geometries") or {}
    rows: dict = {}
    for entries in tables.values():
        for e in entries:
            r = rows.setdefault((e["d"], e["K"]), {"gib": 0.0, "n": 0, "legal": True,
                                                  "bpw": e.get("bits_per_weight")})
            r["gib"] += e.get("stack_gib", 0.0)
            r["n"] += 1
            r["legal"] = r["legal"] and bool(e.get("legal")) and bool(e.get("fused_kernel_ok", True))
    return [Geometry(d, k, r["gib"], r["bpw"]) for (d, k), r in sorted(rows.items())
            if r["n"] == len(tables) and r["legal"]]


def _legal(prof: dict, d: int, k: int) -> Geometry:
    """The profile's legal geometry (d, K), or refused."""
    for g in legal_geometries(prof):
        if (g.d, g.K) == (d, k):
            return g
    raise RecipeError(f"{geo_key(d, k)} is not a legal geometry on every projection of this teacher")


def _affine_gib(gib: float, bits: int, group: int) -> float:
    """`gib` of bf16 weights as mlx affine: bits per weight plus a 16-bit
    scale and bias per group."""
    return gib * (bits + 32 / group) / 16


def _convert_writes_gib(by: dict, bits: int) -> float:
    """What a stream-convert at `bits` writes, as its preflight states it."""
    return round(by["carry"] + by["embed"] + by["vq_target"] * bits / 16, 1)


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


def _int(v, what: str) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, str)):
        raise RecipeError(f"{what} must be an integer, not {v!r}")
    try:
        return int(v)
    except ValueError:
        raise RecipeError(f"{what} must be an integer, not {v!r}")


def _layers(prof: dict) -> tuple[list[int], str]:
    """(the VQ layers, sorted; fit-moe's --vq-layers for exactly them: "a-b"
    when contiguous, else the comma list it also accepts)."""
    vq = sorted({_int(i, "a profile vq_layers entry") for i in prof["arch"].get("vq_layers") or []})
    if not vq:
        raise RecipeError("the profile lists no VQ layers")
    if vq == list(range(vq[0], vq[-1] + 1)):
        return vq, f"{vq[0]}-{vq[-1]}"
    return vq, ",".join(map(str, vq))


@dataclasses.dataclass(frozen=True)
class Recipe:
    """A recipe, validated as far as it can be without the teacher's profile
    (which geometry is legal, which layers exist: the compile checks those)."""
    raw: dict                       # as given; recorded in the queue file
    name: str
    family: str
    teacher: str
    mode: str                       # vq | affine
    bits: int
    group: int
    geo: tuple[int, int] | None     # vq only
    tail: tuple[int, int, int] | None   # MoE vq only: (from layer, d, K)
    protect_bits: int               # MoE vq only
    base: str | None                # dense: the quantized incumbent
    vision: bool
    score: bool
    mtp: bool

    @property
    def work(self) -> str:
        return f"<scratch>/{self.name}"

    @property
    def out(self) -> str:
        return f"{self.work}/{self.name}"

    @classmethod
    def from_json(cls, r) -> "Recipe":
        if not isinstance(r, dict):
            raise RecipeError("a recipe is a JSON object")
        name, family, teacher = r.get("name", ""), r.get("family", ""), r.get("teacher", "")
        if not NAME.match(name):
            raise RecipeError("name: lowercase letters, digits, '.', '_' or '-', at most 80")
        mode = r.get("mode", "vq")
        if mode not in ("vq", "affine"):
            raise RecipeError(f"mode {mode!r}: vq or affine")
        if family not in MOE | DENSE:
            raise RecipeError(f"no build of family {family!r} has run through a queue yet; "
                              f"compiled families: {', '.join(sorted(MOE | DENSE))}")
        bits, group = _int(r.get("bits", 4), "bits"), _int(r.get("group", 64), "group")
        geo = geometry(r.get("geo")) if mode == "vq" else None
        tail, pb = None, 8
        if mode == "vq" and family in MOE:
            pb = _int(r.get("protect_bits", 8), "protect_bits")
            if not 2 <= pb <= 8:
                raise RecipeError("protect_bits (attention, embeddings, shared modules) must be 2-8")
            t = r.get("tail")
            if t is not None and not isinstance(t, dict):
                raise RecipeError('tail is {"from": layer, "geo": "dD-KK"}')
            if t:
                tail = (_int(t.get("from", -1), "tail.from"), *geometry(t.get("geo")))
        return cls(raw=r, name=name, family=family, teacher=teacher, mode=mode, bits=bits,
                   group=group, geo=geo, tail=tail, protect_bits=pb, base=r.get("base") or None,
                   vision=bool(r.get("vision")), score=bool(r.get("score")), mtp=bool(r.get("mtp")))


# Each chain builder -> (steps, artifact GiB, scratch GiB, warnings); `vis` is
# the vision tower's GiB when the recipe grafts one, else 0.
def _affine_steps(r: Recipe, prof: dict, src: str, by: dict, vis: float):
    art_gib = _affine_gib(by["carry"] + by["embed"] + by["vq_target"], r.bits, r.group) + vis
    steps = [_step("convert", "stream-convert",
                   ["--src", src, "--out", r.out, "--bits", r.bits, "--family", r.family],
                   {"full": "stream-convert has no partial mode",
                    "writes_gib": _convert_writes_gib(by, r.bits)},
                   retries=1, timeout_s=6 * 3600, expect={"files": [f"{r.out}/config.json"]})]
    return steps, art_gib, art_gib, []


def _moe_steps(r: Recipe, prof: dict, src: str, by: dict, vis: float):
    d, k = r.geo
    g = _legal(prof, d, k)
    vq, spec = _layers(prof)
    lo, hi = vq[0], vq[-1]
    struct = f"{r.work}/struct"
    skel_gib = _affine_gib(by["carry"] + by["embed"], r.protect_bits, 64)
    struct_gib = round(skel_gib + _affine_gib(by["vq_target"], 2, 64), 1)   # experts: 2-bit markers
    steps, warns = [], []
    steps.append(_step("struct-base", "stream-convert",
                       ["--src", src, "--out", struct, "--struct", "--protect-bits", r.protect_bits,
                        "--family", r.family],
                       {"full": "the fit needs the whole struct base", "writes_gib": struct_gib},
                       retries=1, timeout_s=6 * 3600, expect={"files": [f"{struct}/config.json"]}))
    args = ["--base", struct, "--src", src, "--out", r.out, "--vq-layers", spec,
            "--dim", d, "--k", k, "--family", r.family, "--seed", 1234, "--pack"]
    abort = relerr_abort(d, k)
    tg, ntail = None, 0
    if r.tail:
        frm, td, tk = r.tail
        tg = _legal(prof, td, tk)
        if not lo < frm <= hi or frm not in vq:
            raise RecipeError(f"tail.from {frm} must be a VQ layer in {lo + 1}-{hi}")
        args += ["--tail-from", frm, "--tail-geom", f"d{td}k{tk}"]     # fit-moe's own spelling
        if relerr_abort(td, tk) != abort:
            warns.append(f"fit-moe takes one --relerr-abort: the looser of the two geometries' "
                         f"({max(abort, relerr_abort(td, tk))}) applies to every layer")
        abort = max(abort, relerr_abort(td, tk))
        ntail = sum(i >= frm for i in vq)
    n = len(vq)
    stack = g.gib * (n - ntail) / n + (tg.gib * ntail / n if tg else 0)
    args += ["--relerr-abort", abort]
    art_gib = skel_gib + stack + vis
    one = f"{lo},{r.tail[0]}" if r.tail else f"{lo}"
    i = args.index("--vq-layers")
    steps.append(_step("fit", "fit-moe", args,
                       {"args": [*args[:i], "--vq-layers", one, *args[i + 2:]],
                        # fit-moe rewrites every base shard, whatever --vq-layers says
                        "writes_gib": struct_gib},
                       retries=3, timeout_s=48 * 3600, expect={"files": [f"{r.out}/config.json"]}))
    return steps, art_gib, struct_gib + art_gib, warns


def _dense_steps(r: Recipe, prof: dict, src: str, by: dict, vis: float):
    d, k = r.geo
    g = _legal(prof, d, k)
    vq, spec = _layers(prof)
    lo, hi = vq[0], vq[-1]
    if "," in spec:          # fit-dense --layers takes a range only
        raise RecipeError(f"the profile's VQ layers are not contiguous ({spec}); "
                          "fit-dense takes one range")
    fit, build = f"{r.work}/fit", f"{r.work}/build"
    base_gib = _affine_gib(by["carry"] + by["embed"], r.bits, r.group)
    art_gib = base_gib + g.gib + vis
    scratch_gib = 2 * art_gib + (0 if r.base else base_gib + _affine_gib(by["vq_target"], r.bits, r.group))
    steps, warns = [], []
    if r.base:               # the quantized incumbent, as the shipped dense build used
        base = source_path(r.base)
    else:
        base = f"{r.work}/base-q{r.bits}"
        steps.append(_step("base", "stream-convert",
                           ["--src", src, "--out", base, "--bits", r.bits, "--family", r.family],
                           {"full": "the build splices into the whole base",
                            "writes_gib": _convert_writes_gib(by, r.bits)},
                           retries=1, timeout_s=6 * 3600, expect={"files": [f"{base}/config.json"]}))
        warns.append("no incumbent base given: the base is made by stream-convert, and the "
                     "build checks every fitted module lands on it (build-dense --dry-run)")
    fargs = ["--src", src, "--out", fit, "--family", r.family, "--layers", f"{lo}-{hi}",
             "--dim", d, "--k", k, "--group", r.group, "--seed", 1234]
    i = fargs.index("--layers")
    steps.append(_step("fit", "fit-dense", fargs,
                       {"args": [*fargs[:i], "--layers", f"{lo}-{lo}", *fargs[i + 2:]]},
                       retries=3, timeout_s=24 * 3600))
    steps.append(_step("remap-check", "build-dense",
                       ["--family", r.family, "--base", base, "--mlp", fit, "--out", build, "--dry-run"],
                       {"full": "validates names only, writes nothing", "writes_gib": 0},
                       retries=0, timeout_s=1800, gpu=False))
    steps.append(_step("build", "build-dense",
                       ["--family", r.family, "--base", base, "--mlp", fit, "--out", build],
                       {"full": "assembles what the fit wrote", "writes_gib": round(sum(by.values()), 1)},
                       retries=1, timeout_s=3 * 3600))
    steps.append(_step("pack", "pack-dense", ["--src", build, "--out", r.out],
                       {"full": "packs what the build wrote", "writes_gib": round(sum(by.values()), 1)},
                       retries=1, timeout_s=3 * 3600, expect={"files": [f"{r.out}/config.json"]}))
    return steps, art_gib, scratch_gib, warns


def _tail_steps(r: Recipe, src: str, by: dict):
    """What every chain ends with: [vision graft + smoke], the pin's smoke,
    the bundle and release checks, [kl-ladder] -> (steps, warnings)."""
    out, steps, warns = r.out, [], []
    if r.vision:
        if by.get("vision"):
            steps.append(_step("vision", "graft", ["--artifact", out, "--src", src, *_graft_flags(src)],
                               {"full": "grafts the tower in place", "writes_gib": round(by["vision"] + 0.5, 1)},
                               retries=0, timeout_s=3600))
            steps.append(_step("vision-smoke", "vision-smoke", [out, "--max-tokens", 8],
                               {"full": "reads the artifact, writes nothing", "writes_gib": 0},
                               retries=0, timeout_s=3600))
        else:
            warns.append("this teacher has no vision tower; nothing to graft")
    if r.mode == "vq" and r.mtp:
        warns.append("the MTP head is not compiled: choose its wiring with `vqlab mtp-probe35`, "
                     "then `vqlab mtp-pack` by hand")
    pin = f"{r.work}/pin"
    steps.append(_step("smoke", "pin", [out, "--out", pin, "--max-tokens", 8],
                       {"full": "a pin is symlinks plus an 8-token smoke", "writes_gib": 0},
                       retries=0, timeout_s=3600))
    steps.append(_step("runtime-check", "check-bundle", ["--artifact", out, "--strict"],
                       {"full": "a read-only check", "writes_gib": 0}, retries=0, timeout_s=600))
    steps.append(_step("release-check", "check-release",
                       ["--artifact", out, "--tokenizer-ref", f"{src}/tokenizer.json"],
                       {"full": "reads the artifact; its smokes are short", "writes_gib": 0},
                       retries=0, timeout_s=2 * 3600))
    if r.score:
        caches = _caches(r.teacher)
        if caches:
            kl = [a for c in caches for a in ("--cache", c)] + [
                "--rung", f"{r.name}={pin}", "--per-pos-dir", f"{r.work}/kl-per-pos",
                "--out", f"{r.work}/kl.json", "--lazy-over-gb", 16]
            warns.append("the score is this build's absolute KL against the teacher; the release "
                         "gate is a PAIRED kl-ladder against the incumbent rung (FINDINGS: KL is the gate)")
            steps.append(_step("score", "kl-ladder", kl, {"append": ["--preflight"]},
                               retries=1, timeout_s=12 * 3600,
                               expect={"stdout_regex": "KL", "files": [f"{r.work}/kl.json"]}))
        else:
            warns.append(f"no KL caches are registered for {r.teacher}: not scored "
                         "(build them with `vqlab stream-score`, then register them)")
    return steps, warns


def compile_recipe(r: dict) -> tuple[dict, list[str]]:
    """-> (queue file dict, warnings). Raises RecipeError, and only that:
    a recipe or profile missing a key it needs is refused, not a crash."""
    try:
        return _compile(r)
    except RecipeError:
        raise
    except (KeyError, TypeError, AttributeError, ValueError) as e:
        raise RecipeError(f"the recipe or the teacher's profile is malformed ({type(e).__name__}: {e})")


def _compile(raw: dict) -> tuple[dict, list[str]]:
    r = Recipe.from_json(raw)
    prof = load_profile(r.family, r.teacher)
    src = source_path(prof["identity"]["path"])
    kind = prof["arch"].get("kind")
    if (r.family in MOE) != (kind == "moe"):
        raise RecipeError(f"{r.teacher} is {kind}, but {r.family} builds are compiled as "
                          f"{'moe' if r.family in MOE else 'dense'}")
    by = prof["bytes"]["by_class_gib"]
    vis = by.get("vision", 0.0) if r.vision else 0.0
    chain = _affine_steps if r.mode == "affine" else _moe_steps if r.family in MOE else _dense_steps
    steps, art_gib, scratch_gib, warns = chain(r, prof, src, by, vis)
    tail, tail_warns = _tail_steps(r, src, by)
    return {"name": r.name, "commit": "HEAD", "recipe": r.raw,
            # estimates for the confirmation, not measurements: the artifact
            # from the profile's exact per-geometry bytes, scratch = every
            # intermediate the chain leaves (struct base, unpacked build)
            "estimate": {"artifact_gib": round(art_gib, 2), "scratch_gib": round(scratch_gib, 1)},
            "steps": steps + tail}, warns + tail_warns


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
