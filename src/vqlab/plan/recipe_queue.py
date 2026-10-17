"""vqlab recipe — compile a build recipe into a `vqlab queue` file.

    vqlab recipe compile recipe.json [--out queue.json]

A recipe says WHAT to build; this writes HOW, as the steps a queue runs
(queue/runner.py): pinned code, the GPU lease, preflight, loud failure.
Every step is a vqlab command with the flags the lab's queues of record ran
it with; a chain is offered only for families whose pieces have each run:

  MoE (qwen3_5, qwen4_exp, glm5_next):  stream-convert --struct -> fit-moe
                  --pack [--tail-from/--tail-geom: a mixed build]
                  [-> fit-ple -> splice-ple -> pack-ple: a family with n-gram
                  tables, Flash-Next; "ple_geo" picks their geometry]
  dense (qwen3_8_dense, gemma4_e4b):  [stream-convert --bits: the base,
                  unless an incumbent is given] -> fit-dense -> build-dense
                  --dry-run -> build-dense -> pack-dense
                  [a family whose VQ target is one embedding table, e4b: its
                  incumbent Hub build is the base, fit-ple of that table in place
                  of fit-dense, build-dense --embed]
  affine (either):  stream-convert --bits [--group] [--protect-bits: MoE
                  mixed affine, experts at bits/group, the rest at protect/64]

and every chain, whatever its mode, ends the same way: [graft ->
vision-smoke] -> [sz-pack -> sz-check -> bundle] -> [mtp-extract -> mtp-pack ->
mtp-smoke-head -> mtp-accept] -> pin -> check-bundle --strict -> check-release ->
[kl-ladder].

A build whose estimated size is past THIS box's resident bar (0.90 x its RAM,
read from the machine at compile time: preflight_ram.bar_gib) cannot be loaded
here, so its smokes go to the cluster through Knurlogic, the way the published
397B builds were gated: the local vision-smoke is dropped, knurlogic-link makes
the build loadable by name (one symlink, no copy), and check-release runs with
--knurlogic-smoke fit (text, then the image through the same runtime). Its
preflight is full-size and real, so a cluster that is down fails it loudly.

"skipzero" (MoE VQ only) drops the teacher's dead expert rows on disk the way
every published Qwen3.5-397B build was made (docs/SKIPZERO.md, F175): sz-pack
of the finished, grafted artifact, then sz-check, then `bundle` so the runtime
serves the packed copy, and the pin and every check after it run on that
copy. It is offered only for a teacher whose dead
rows were measured (families/<f>/teachers/<t>/skipzero.json), and the size it
saves comes from those counts at the recipe's geometry (skipzero_gib).

"mtp" packs the teacher's own draft head the way every published sidecar
was made: mtp-extract of the head's bf16 tensors, then mtp-pack at its
default recipe (6-bit, group 32, mtp.registry.PACK_*; "mtp_bits" picks 4, 5, 6
or 8) into the finished artifact as mtp-head-q6.safetensors (mtp-head-q4... at
another width), with the wiring the checkpoint does not determine (fc_order,
h_source for qwen3_5) taken from the teacher's record of
what shipped (families/<f>/teachers/<t>/mtp.json). Where none is recorded the
queue measures it: a trial pack, the family's probe (`mtp-probe35`), and the
real pack reads the probe's choice (--wiring-from); a wrong guess drafts at
chance with no error, so the compiler never makes one. mtp-smoke-head then
proves the sidecar loads and runs, and mtp-accept fails the queue on a dead
head; past the resident bar check-release does that through Knurlogic,
which serves the build with drafting on. A teacher with no head, or a model
type vqlab packs no head for, is refused.

A compiled chain is proven by its PREFLIGHT, which runs every step small
and real (the fit on one layer, plus the tail's first layer for a mixed
build); start the full run only from a preflight that passed. Which family
builds which chain, and what it does differently (n-gram tables, a tower
that rides the base, a head stored as a layer), is its family registry
entry (family/registry.py "recipe"); any other family is refused by name.
kl-ladder is added only when the teacher's caches are registered
(families/<f>/teachers/<t>/teacher_caches.json); it scores absolute KL, not
the paired release gate. A fit's --relerr-abort comes from measured healthy
ranges only (FINDINGS I.8); an unmeasured geometry is refused.

Recipe (JSON):

    {"name": "q35-d4k256", "family": "qwen3_5", "teacher": "Qwen--Qwen3.6-35B-A3B-bf16",
     "mode": "vq", "geo": "d4-K256",
     "tail": {"from": 30, "geo": "d4-K512"},      optional: a mixed build
     "bits": 4, "group": 64,                       affine (and the dense base)
     "protect_bits": 8,                            MoE: attention/embeddings/shared (struct base;
                                                   affine: optional, absent = uniform)
     "base": "<models>/org--name-4bit",            dense: the quantized incumbent
     "vision": true, "score": true, "mtp": true,   mtp: the teacher's draft head, packed
     "mtp_bits": 6,                                the head's width: 4, 5, 6 (default) or 8
     "skipzero": true}                             MoE vq: dead rows dropped on disk

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
from vqlab.mtp import registry as mtp_registry
from vqlab.plan import preflight_ram
from vqlab.util.errors import UserError
from vqlab.util.geo import geo_key
from vqlab.core.artifacts.artifact import Artifact, INDEX
from vqlab.core.source import teacher_profiles
from vqlab.util.fs import GIB, write_text_atomic

# the families each build chain serves, as the family registry declares them
MOE = registry.recipe_families("moe")
DENSE = registry.recipe_families("dense")
# every family the compiler builds
COMPILED = MOE | DENSE
GEO = re.compile(r"^d(\d+)-?[kK](\d+)$")
NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,79}$")
# the codebook dims sz-pack packs (skipzero_load.SUPPORTED_DIMS, a peer layer;
# tests/test_recipe_queue.py holds the two equal)
SZ_DIMS = (2, 4, 8)
# mtp-accept's gate: below this pooled acceptance the head is dead. The
# published heads read 0.63-0.90 (docs/MTP.md), a wrong wiring ~0.00.
MTP_ACCEPT_FLOOR = 0.3
# the n-gram (PLE) table geometries the published Flash-Next builds shipped
# (their cards: d8/K256 on 2.1bpw, d4/K2048 on 3.2bpw, d8/K4096 on 4.4/5.5bpw);
# the default is the 3.2bpw build's
PLE_GEOS = ("d4-K2048", "d8-K256", "d8-K4096")
PLE_GEO = "d4-K2048"
# rows a one-table fit preflight fits (fit-ple --rows): one of its default
# row chunks, so the preflight's fit is minutes, not the table
TABLE_PREFLIGHT_ROWS = 4096
# tokens per prompt over mtp-accept's 12 prompts: ~700 draft steps, minutes
# on a resident build, and far more than a live/dead verdict needs
MTP_ACCEPT_TOKENS = 64


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


def _rest(by: dict) -> float:
    """GiB of everything a base carries besides the experts, tower and head:
    carry, embeddings and any n-gram tables (the profile's own class)."""
    return by["carry"] + by["embed"] + by.get("ple", 0.0)


def _convert_writes_gib(by: dict, bits: int) -> float:
    """What a stream-convert at `bits` writes, as its preflight states it."""
    return round(_rest(by) + by["vq_target"] * bits / 16, 1)


def _fam(family: str) -> dict:
    """The family's registry entry: what its chain does differently."""
    return registry.FAMILY.get(family) or {}


def _vq_gib(gib: float, d: int, k: int, group: int) -> float:
    """`gib` of bf16 weights as VQ codes plus a 16-bit scale per group."""
    return gib * ((k.bit_length() - 1) / d + 16 / group) / 16


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
    teacher = Artifact(config.expand(src))
    try:
        keys = teacher.index
    except (OSError, ValueError, KeyError) as e:
        raise RecipeError(f"cannot read the teacher's tensor index "
                          f"({teacher.dir / INDEX}): {e}") from None
    if any(k.startswith("model.visual.") for k in keys):
        return ["--dest-prefix", "vision_tower", "--permute-conv5"]
    return []


def _int(v, what: str) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, str)):
        raise RecipeError(f"{what} must be an integer, not {v!r}")
    try:
        return int(v)
    except ValueError:
        raise RecipeError(f"{what} must be an integer, not {v!r}") from None


def _protect_bits(v) -> int:
    pb = _int(v, "protect_bits")
    if not 2 <= pb <= 8:
        raise RecipeError("protect_bits (attention, embeddings, shared modules) must be 2-8")
    return pb


def _layers(prof: dict) -> tuple[list[int], str]:
    """(the VQ layers, sorted; fit-moe's --vq-layers for exactly them: "a-b"
    when contiguous, else the comma list it also accepts)."""
    vq = sorted({_int(i, "a profile vq_layers entry") for i in prof["arch"].get("vq_layers") or []})
    if not vq:
        raise RecipeError("the profile lists no VQ layers")
    if vq == list(range(vq[0], vq[-1] + 1)):
        return vq, f"{vq[0]}-{vq[-1]}"
    return vq, ",".join(map(str, vq))


def skipzero_gib(prof: dict, sz: dict, layer_geo: dict) -> float:
    """GiB that sz-pack drops from a build whose VQ layer L sits at geometry
    layer_geo[L] = (d, K): per module with dead rows, its dead share of rows
    times the module's exact bytes at that geometry (the profile's table),
    less the row mask sz-pack keeps (one bit per row). Dead rows are a
    teacher property, so this is the measured relation, not a fit: against
    the published 397B builds it lands within 0.005 GiB of what sz-pack
    recorded (skipzero.json "checks")."""
    per = {(p, e["d"], e["K"]): e.get("bytes") for p, es in (prof.get("geometries") or {}).items()
           for e in es}
    rows, saved = sz["rows"], 0.0
    for layer, projs in sz["dead_rows"].items():
        geo = layer_geo.get(int(layer))
        if geo is None:
            continue
        if geo[0] not in SZ_DIMS:
            raise RecipeError(f"sz-pack packs dims {list(SZ_DIMS)} only, not {geo_key(*geo)}")
        for proj, dead in projs.items():
            b = per.get((proj, *geo))
            if not b:
                raise RecipeError(f"the profile has no byte count for {proj} at {geo_key(*geo)}")
            if dead:
                saved += dead / rows[proj] * b - rows[proj] / 8
    return saved / GIB


def _skipzero_rows(r: "Recipe") -> dict:
    """The teacher's measured dead rows, or refused: skipzero is compiled
    only where they were measured."""
    sz = registry.teacher_record(r.family, r.teacher, "skipzero.json")
    if not sz or not sz.get("dead_rows"):
        raise RecipeError(f"no dead rows are measured for {r.teacher} "
                          f"(families/{r.family}/teachers/{r.teacher}/skipzero.json): "
                          "skipzero is offered only where sz-pack has run")
    return sz


@dataclasses.dataclass(frozen=True)
class MtpHead:
    """The draft head a recipe packs: its family's MTP spec, the measured
    wiring (None: the queue's probe chooses it), and its GiB as the bf16
    graft and as the packed sidecar (the profile's mtp bytes at
    mtp-pack's recipe at `bits`)."""
    spec: mtp_registry.FamilySpec
    wiring: dict | None
    graft_gib: float
    gib: float
    bits: int
    extract: tuple = ()             # mtp-extract's key flags: a head stored off mtp.*


def mtp_head(family: str, teacher: str, prof: dict,
             bits: int = mtp_registry.PACK_BITS) -> MtpHead:
    """The head `vqlab mtp-pack` builds for this teacher, or refused: a
    teacher whose checkpoint has none, a model type vqlab packs no head for,
    or a recorded wiring that does not name exactly the choices its family
    leaves open."""
    graft = ((prof.get("bytes") or {}).get("by_class_gib") or {}).get("mtp")
    n = (prof.get("arch") or {}).get("mtp_layer")
    # a head stored as upstream layers.<n> (GLM): mtp-extract takes it by that
    # key, as the head module reads it (layers.<n>.*, mtp_head_glm5.load_graft)
    extract = (() if n is None else
               ("--key-regex", rf"\.layers\.{n}\.",
                "--strip-prefix", "model.language_model."))
    if not graft:
        raise RecipeError(f"{teacher} has no MTP head: its profile counts no "
                          "mtp.* tensors")
    mt = (prof.get("arch") or {}).get("model_type")
    spec = mtp_registry.FAMILIES.get(mt)
    if spec is None:
        raise RecipeError(f"vqlab packs no MTP head for model_type {mt!r} "
                          "(none registered in vqlab/mtp/registry.py)")
    if spec.wiring and spec.probe is None:
        raise RecipeError(f"no probe for the {mt!r} head's wiring is registered "
                          "(vqlab/mtp/registry.py), so the compiler cannot choose it")
    record = f"families/{family}/teachers/{teacher}/mtp.json"
    wiring = (registry.teacher_record(family, teacher, "mtp.json") or {}).get("wiring")
    if wiring is not None:
        if not isinstance(wiring, dict) or sorted(wiring) != sorted(spec.wiring):
            raise RecipeError(f"{record}: the wiring must set exactly "
                              f"{', '.join(spec.wiring)}, not {wiring!r}")
        bad = [f"{k}={v!r}" for k, v in wiring.items()
               if v not in mtp_registry.WIRING_CHOICES[k]]
        if bad:
            raise RecipeError(f"{record}: {', '.join(bad)} is not a value mtp-pack takes "
                              f"({mtp_registry.WIRING_CHOICES})")
    if not spec.wiring:
        wiring = {}          # the checkpoint determines this head: nothing to choose
    return MtpHead(spec, wiring, graft,
                   _affine_gib(graft, bits, mtp_registry.PACK_GROUP), bits, extract)


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
    ple: tuple[int, int] | None     # vq, a family with n-gram tables: their (d, K)
    protect_bits: int | None        # MoE: vq struct base (default 8); affine mixed (None = uniform)
    base: str | None                # dense: the quantized incumbent
    vision: bool
    score: bool
    mtp: bool
    mtp_bits: int                   # the draft head's width, one of mtp.registry.PACK_BITS_CHOICES
    skipzero: bool                  # MoE vq only: sz-pack the finished artifact

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
        if family not in COMPILED:
            raise RecipeError(f"no build of family {family!r} has run through a queue yet; "
                              f"compiled families: {', '.join(sorted(COMPILED))}")
        bits = _int(r.get("bits", registry.base_bits(family)), "bits")
        group = _int(r.get("group", 64), "group")
        geo = geometry(r.get("geo")) if mode == "vq" else None
        tail, pb = None, 8
        if mode == "vq" and family in MOE:
            pb = _protect_bits(r.get("protect_bits", 8))
            t = r.get("tail")
            if t is not None and not isinstance(t, dict):
                raise RecipeError('tail is {"from": layer, "geo": "dD-KK"}')
            if t:
                tail = (_int(t.get("from", -1), "tail.from"), *geometry(t.get("geo")))
        if mode == "affine" and r.get("protect_bits") is not None:
            if family not in MOE:
                raise RecipeError("protect_bits on an affine build splits experts from the rest; "
                                  f"{family} has no experts")
            pb = _protect_bits(r["protect_bits"])
        elif mode == "affine":
            pb = None
        if mode == "affine" and group not in (32, 64, 128):
            raise RecipeError(f"group {group}: 32, 64 or 128")
        ple = None
        if mode == "vq" and _fam(family).get("ple"):
            ple = geometry(r.get("ple_geo", PLE_GEO))
            if geo_key(*ple) not in PLE_GEOS:
                raise RecipeError(f"ple_geo {geo_key(*ple)}: the n-gram tables have "
                                  f"shipped at {', '.join(PLE_GEOS)} only")
        sz = r.get("skipzero", False)
        if not isinstance(sz, bool):
            raise RecipeError(f"skipzero must be true or false, not {sz!r}")
        if sz and not (mode == "vq" and family in MOE):
            raise RecipeError("skipzero has run only on MoE VQ builds (qwen3_5, docs/SKIPZERO.md); "
                              f"not on a {family} {mode} build")
        mtp_bits = _int(r.get("mtp_bits", mtp_registry.PACK_BITS), "mtp_bits")
        if mtp_bits not in mtp_registry.PACK_BITS_CHOICES:
            raise RecipeError(f"mtp_bits {mtp_bits}: "
                              f"{', '.join(map(str, mtp_registry.PACK_BITS_CHOICES))}")
        return cls(raw=r, name=name, family=family, teacher=teacher, mode=mode, bits=bits,
                   group=group, geo=geo, tail=tail, ple=ple, protect_bits=pb,
                   base=r.get("base") or None,
                   vision=bool(r.get("vision")), score=bool(r.get("score")),
                   mtp=bool(r.get("mtp")), mtp_bits=mtp_bits,
                   skipzero=sz)


# Each chain builder -> (steps, artifact GiB, scratch GiB, warnings); `vis` is
# the vision tower's GiB when the recipe grafts one, else 0.
def _affine_steps(r: Recipe, src: str, by: dict, vis: float):
    if _fam(r.family).get("incumbent"):     # its converter would drop the towers it ships
        raise RecipeError(f"this model's affine build is {_fam(r.family)['incumbent']}; "
                          "download it instead")
    if r.protect_bits is None:       # uniform: stream-convert's group applies everywhere
        art_gib = _affine_gib(_rest(by) + by["vq_target"], r.bits, r.group) + vis
    else:                            # mixed: experts bits/group, the rest protect/64
        art_gib = (_affine_gib(by["vq_target"], r.bits, r.group)
                   + _affine_gib(_rest(by), r.protect_bits, 64) + vis)
    args = ["--src", src, "--out", r.out, "--bits", r.bits, "--family", r.family]
    if r.group != 64:
        args += ["--group", r.group]
    if r.protect_bits is not None:
        args += ["--protect-bits", r.protect_bits]
    if r.family in MOE:       # a cold multi-GB read in a GPU buffer trips the watchdog (397B)
        args += ["--cpu-load"]
    steps = [_step("convert", "stream-convert", args,
                   {"full": "stream-convert has no partial mode",
                    "writes_gib": _convert_writes_gib(by, r.bits)},
                   retries=1, timeout_s=6 * 3600, expect={"files": [f"{r.out}/config.json"]})]
    return steps, art_gib, art_gib, art_gib, []


def _moe_steps(r: Recipe, prof: dict, src: str, by: dict, vis: float):
    d, k = r.geo
    g = _legal(prof, d, k)
    vq, spec = _layers(prof)
    lo, hi = vq[0], vq[-1]
    struct = f"{r.work}/struct"
    # n-gram tables: stream-convert writes them at group 32
    ple_gib = by.get("ple", 0.0)
    if r.ple and not ple_gib:
        raise RecipeError(f"{r.teacher}'s profile counts no n-gram tables; "
                          "re-profile it (`vqlab family-profile`)")
    skel_gib = (_affine_gib(by["carry"] + by["embed"], r.protect_bits, 64)
                + _affine_gib(ple_gib, r.protect_bits, 32))
    struct_gib = round(skel_gib + _affine_gib(by["vq_target"], 2, 64), 1)   # experts: 2-bit markers
    steps, warns = [], []
    steps.append(_step("struct-base", "stream-convert",
                       ["--src", src, "--out", struct, "--struct", "--protect-bits", r.protect_bits,
                        "--family", r.family, "--cpu-load"],
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
        tail_abort = relerr_abort(td, tk)
        if tail_abort != abort:
            warns.append(f"The two geometries have different fit-error limits; the looser, "
                         f"{max(abort, tail_abort)}, applies to every layer.")
        abort = max(abort, tail_abort)
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
    if r.ple:
        ple_steps, ple_delta = _ple_steps(r, src, ple_gib)
        steps += ple_steps
        art_gib += ple_delta
    # its preflight fits one layer into the struct base: the base's size
    return steps, art_gib, struct_gib + art_gib, struct_gib + vis, warns


def _ple_steps(r: Recipe, src: str, gib: float):
    """The n-gram tables, as the published Flash-Next builds made them: fit
    on their own (fit-ple), spliced into the packed build in place of their
    affine copies (splice-ple), then packed to their true code width
    (pack-ple) -> (steps, the artifact's GiB change)."""
    ple = _fam(r.family)["ple"]
    (d, k), fit = r.ple, f"{r.work}/ple-fit"
    codes = _vq_gib(gib, d, k, ple["group"])
    args = ["--src", src, "--out", fit, "--key-regex", ple["key_regex"],
            "--dim", d, "--k", k, "--group", ple["group"], "--seed", 1234]
    steps = [
        _step("ple-fit", "fit-ple", args, {"append": ["--limit", 1]},
              retries=3, timeout_s=24 * 3600),
        _step("ple-splice", "splice-ple", ["--artifact", r.out, "--ple-fit", fit],
              {"full": "splices what the fit wrote", "writes_gib": round(codes, 1)},
              retries=0, timeout_s=3 * 3600),
        _step("ple-pack", "pack-ple", ["--artifact", r.out],
              {"full": "packs what the splice wrote", "writes_gib": round(codes, 1)},
              retries=0, timeout_s=3 * 3600)]
    # the struct base carried the tables at the protect width, group 32
    return steps, codes - _affine_gib(gib, r.protect_bits, 32)


def incumbent(family: str) -> tuple[str, float] | None:
    """A one-table family's base: its incumbent Hub build (registry
    "incumbent") in recorded form, and the GiB the artifact keeps of it --
    its shard bytes less the table's affine tensors, read from the headers,
    so towers and every module ride at the widths it ships. None for a
    family without one; refused when it is not on this machine."""
    fam = _fam(family)
    repo, emb = fam.get("incumbent"), fam.get("embed")
    if not (repo and emb):
        return None
    path = teacher_profiles.repo_source(repo)
    if path is None:
        raise RecipeError(f"download {repo} first: it is the base this model is built on")
    art = Artifact(config.expand(path))
    keep = sum(v["data_offsets"][1] - v["data_offsets"][0]
               for f in art.shards for k, v in art.header(f).items()
               if not k.startswith(emb["key"] + "."))
    return path, keep / GIB


def _dense_fit(r: Recipe, prof: dict, src: str, fit: str):
    """The dense fit -> (its step, build-dense's flag for what it wrote):
    fit-dense of the MLP layers, or, for a family whose VQ target is one
    embedding table (registry "embed"), fit-ple of that table as the
    shipped e4b build was made; its preflight fits a few rows of it."""
    d, k = r.geo
    table = registry.embed_regex(r.family)
    if table:
        args = ["--src", src, "--out", fit, "--key-regex", table, "--dim", d, "--k", k,
                "--group", _fam(r.family)["embed"]["group"], "--seed", 1234]
        return _step("fit", "fit-ple", args,
                     {"append": ["--rows", TABLE_PREFLIGHT_ROWS]},
                     retries=3, timeout_s=24 * 3600), "--embed"
    vq, spec = _layers(prof)
    lo, hi = vq[0], vq[-1]
    if "," in spec:          # fit-dense --layers takes a range only
        raise RecipeError(f"the profile's VQ layers are not contiguous ({spec}); "
                          "fit-dense takes one range")
    args = ["--src", src, "--out", fit, "--family", r.family, "--layers", f"{lo}-{hi}",
            "--dim", d, "--k", k, "--group", r.group, "--seed", 1234]
    i = args.index("--layers")
    return _step("fit", "fit-dense", args,
                 {"args": [*args[:i], "--layers", f"{lo}-{lo}", *args[i + 2:]]},
                 retries=3, timeout_s=24 * 3600), "--mlp"


def _measured_gib(base: str) -> float:
    """GiB of a given base's weights, read from its shards on this machine."""
    local = pathlib.Path(config.expand(base))
    if not (local / INDEX).is_file():
        raise RecipeError("the base you gave is not a model: it has no weight index")
    return Artifact(local).size() / GIB


def _dense_steps(r: Recipe, prof: dict, src: str, by: dict, vis: float):
    g = _legal(prof, *r.geo)
    fit, build = f"{r.work}/fit", f"{r.work}/build"
    fit_step, fit_flag = _dense_fit(r, prof, src, fit)
    inc = None if r.base else incumbent(r.family)
    base_gib = _affine_gib(_rest(by), r.bits, r.group)
    # an incumbent as shipped, towers and all, is measured, not modelled
    art_gib = (inc[1] if inc else base_gib + vis) + g.gib
    converts = not (r.base or inc)
    whole_gib = base_gib + _affine_gib(by["vq_target"], r.bits, r.group)
    scratch_gib = 2 * art_gib + (whole_gib if converts else 0)
    steps, warns = [], []
    # the quantized incumbent, as the shipped dense builds used, else one made here
    base = (source_path(r.base) if r.base else inc[0] if inc
            else f"{r.work}/base-q{r.bits}")
    # its preflight fits one layer (a one-table family: a few rows of the
    # table, at full shape) into the base, so it is the base as it stands
    pf_gib = (art_gib if inc                    # an incumbent's preflight is its build
              else _measured_gib(base) + vis if r.base
              else whole_gib + vis)             # a converted base: every MLP still affine
    if converts:
        steps.append(_step("base", "stream-convert",
                           ["--src", src, "--out", base, "--bits", r.bits, "--family", r.family],
                           {"full": "the build splices into the whole base",
                            "writes_gib": _convert_writes_gib(by, r.bits)},
                           retries=1, timeout_s=6 * 3600, expect={"files": [f"{base}/config.json"]}))
        warns.append("No existing base was given, so the build makes its own affine base first.")
    steps.append(fit_step)
    steps.append(_step("remap-check", "build-dense",
                       ["--family", r.family, "--base", base, fit_flag, fit, "--out", build, "--dry-run"],
                       {"full": "validates names only, writes nothing", "writes_gib": 0},
                       retries=0, timeout_s=1800, gpu=False))
    steps.append(_step("build", "build-dense",
                       ["--family", r.family, "--base", base, fit_flag, fit, "--out", build],
                       {"full": "assembles what the fit wrote", "writes_gib": round(sum(by.values()), 1)},
                       retries=1, timeout_s=3 * 3600))
    steps.append(_step("pack", "pack-dense", ["--src", build, "--out", r.out],
                       {"full": "packs what the build wrote", "writes_gib": round(sum(by.values()), 1)},
                       retries=1, timeout_s=3 * 3600, expect={"files": [f"{r.out}/config.json"]}))
    return steps, art_gib, scratch_gib, pf_gib, warns


def _mtp_steps(r: Recipe, head: MtpHead, src: str, out: str, oversized: bool):
    """The draft head, packed into the finished artifact `out` and gated ->
    (steps, warnings). Its sidecar sits in `out` before the pin and before
    knurlogic-link, so the folder Knurlogic serves carries it."""
    name = head.spec.sidecar_for(head.bits)
    graft, sidecar = f"{r.work}/mtp-graft.safetensors", f"{out}/{name}"
    steps, warns = [], []
    steps.append(_step("mtp-extract", "mtp-extract",
                       ["--src", src, "--out", graft, *head.extract],
                       {"full": "the pack reads the whole head",
                        "writes_gib": round(head.graft_gib, 1)},
                       retries=1, timeout_s=2 * 3600))
    pack = ["--model", out, "--mtp", graft, "--bits", head.bits]
    pack_pf = {"full": "packs the one head", "writes_gib": round(head.gib + 0.5, 1)}
    record = f"families/{r.family}/teachers/{r.teacher}/mtp.json"
    if head.wiring is not None:
        pack += [a for k in head.spec.wiring
                 for a in (f"--{k.replace('_', '-')}", head.wiring[k])]
    elif oversized:
        raise RecipeError(f"no MTP wiring is recorded for {r.teacher} ({record}), and "
                          "its probe needs the trunk resident, which this build is "
                          "too large for here: measure it on a smaller build first")
    else:
        trial, probe = f"{r.work}/mtp-trial/{name}", f"{r.work}/mtp-probe.json"
        steps.append(_step("mtp-trial-pack", "mtp-pack", [*pack, "--out", trial],
                           pack_pf, retries=0, timeout_s=3600))
        steps.append(_step("mtp-wiring-probe", head.spec.probe,
                           ["--model", out, "--sidecar", trial, "--corpus", "lit",
                            "--out", probe],
                           # a preflight's trunk is one fitted layer: its control fails
                           {"append": ["--tokens", 32, "--control-min", 0]},
                           retries=0, timeout_s=2 * 3600))
        pack += ["--wiring-from", probe]
        warns.append("This teacher has no recorded draft-head wiring; the queue measures it "
                     "before packing the head.")
    steps.append(_step("mtp-pack", "mtp-pack", [*pack, "--out", sidecar], pack_pf,
                       retries=0, timeout_s=3600, expect={"files": [sidecar]}))
    steps.append(_step("mtp-smoke", "mtp-smoke-head",
                       ["--model", out, "--sidecar", sidecar],
                       {"append": ["--steps", 2, "--warmup", 1]},
                       retries=0, timeout_s=1800))
    if oversized:
        warns.append("The draft head is checked in the release test; a head that never "
                     "accepts fails the queue.")
    else:
        steps.append(_step("mtp-accept", "mtp-accept",
                           ["--model", out, "--head", f"head={sidecar}",
                            "--tokens", MTP_ACCEPT_TOKENS,
                            "--min-acceptance", MTP_ACCEPT_FLOOR,
                            "--out", f"{r.work}/mtp-accept.json"],
                           # a preflight's trunk is one fitted layer: chance drafts
                           {"append": ["--tokens", 4, "--min-acceptance", 0]},
                           retries=0, timeout_s=2 * 3600))
    return steps, warns


def _vision_steps(r: Recipe, src: str, by: dict, oversized: bool):
    """The tower: grafted into the build and smoked, or, for a family whose
    tower rides the struct base (GLM), only smoked -> (steps, warnings).
    Past the resident bar the release check tests the image through
    Knurlogic instead of the local smoke."""
    if not by.get("vision"):
        return [], (["This teacher has no vision tower to graft."] if r.vision else [])
    steps, warns = [], []
    if _fam(r.family).get("tower") == "base":
        if not r.vision:
            warns.append("This model keeps its vision tower inside the base, so it is "
                         "always included.")
    elif r.vision:
        steps.append(_step("vision", "graft",
                           ["--artifact", r.out, "--src", src, *_graft_flags(src)],
                           {"full": "grafts the tower in place",
                            "writes_gib": round(by["vision"] + 0.5, 1)},
                           retries=0, timeout_s=3600))
    else:
        return [], []
    if not oversized:
        steps.append(_step("vision-smoke", "vision-smoke", [r.out, "--max-tokens", 8],
                           {"full": "reads the artifact, writes nothing",
                            "writes_gib": 0},
                           retries=0, timeout_s=3600))
    return steps, warns


def _tail_steps(r: Recipe, src: str, by: dict, head: MtpHead | None,
                oversized: bool = False):
    """What every chain ends with: [vision graft + smoke], [skip-zero],
    [the MTP head], the pin's smoke, the bundle and release checks,
    [kl-ladder] -> (steps, warnings).

    `oversized`: the artifact is past this box's resident bar
    (preflight_ram), so no load of it can run here. Its text and image smokes
    then run on the cluster through Knurlogic, as the published 397B builds
    were gated: knurlogic-link makes the build loadable by name (one symlink,
    no copy), and check-release --knurlogic-smoke fit loads it where it fits,
    generates, and puts the probe image through the same serving runtime
    (check-release passes its Knurlogic roles to vision-smoke). The local
    vision-smoke after the graft is dropped: it would only refuse."""
    out = r.out
    steps, warns = _vision_steps(r, src, by, oversized)
    if r.skipzero:
        # as the published 397B builds were made: sz-pack of the finished,
        # grafted artifact (config vq_skipzero.source; lab/queues/
        # 2026-09-28-skipzero-397b-full-kl.json for the step shapes)
        sz = f"{r.work}/{r.name}-sz"
        steps.append(_step("skipzero", "sz-pack", [out, "--out", sz],
                           {"append": ["--layers", "0-0"]}, gpu=False,
                           retries=0, timeout_s=6 * 3600, expect={"files": [f"{sz}/config.json"]}))
        steps.append(_step("skipzero-check", "sz-check", [sz, out],
                           {"full": "numpy check, reads only", "writes_gib": 0}, gpu=False,
                           retries=0, timeout_s=3 * 3600))
        # sz-pack leaves the stage-1 experimental hook in model.py
        # (vq_skipzero.loader unset); every published 397B build was then
        # rebundled so the runtime serves it (their provenance histories:
        # sz-pack -> bundle --artifact <the packed copy> -> pin). With no
        # --runtime, bundle keeps the profile the fit baked in.
        steps.append(_step("skipzero-bundle", "bundle", ["--artifact", sz],
                           {"full": "rewrites model.py and config.json, no weights", "writes_gib": 0},
                           retries=0, timeout_s=1800))
        warns.append("Skip-zero is experimental. The packed copy shares its unchanged files "
                     "with the unpacked build, so keep both.")
        out = sz
    if head:
        mtp_steps, mtp_warns = _mtp_steps(r, head, src, out, oversized)
        steps += mtp_steps
        warns += mtp_warns
    pin = f"{r.work}/pin"
    steps.append(_step("smoke", "pin", [out, "--out", pin, "--max-tokens", 8],
                       {"full": "a pin is symlinks plus an 8-token smoke", "writes_gib": 0},
                       retries=0, timeout_s=3600))
    steps.append(_step("runtime-check", "check-bundle", ["--artifact", out, "--strict"],
                       {"full": "a read-only check", "writes_gib": 0}, retries=0, timeout_s=600))
    release = ["--artifact", out, "--tokenizer-ref", f"{src}/tokenizer.json"]
    release_pf = {"full": "reads the artifact; its smokes are short", "writes_gib": 0}
    if oversized:
        steps.append(_step("knurlogic-link", "knurlogic-link", [out],
                           {"full": "one symlink in the Knurlogic models dir, no data", "writes_gib": 0},
                           gpu=False, retries=0, timeout_s=600))
        release += ["--knurlogic-smoke", "fit"]
        # a full preflight: the smokes really load the build through Knurlogic,
        # so a cluster that is down FAILS the step with Knurlogic's own reason
        release_pf = {"full": "loads the artifact on the cluster through Knurlogic; "
                              "fails if Knurlogic or the cluster is down", "writes_gib": 0}
        warns.append("Too large to test on this machine. The release test runs on a machine "
                     "or cluster with more memory, which must be available for the "
                     "preflight and the run.")
    steps.append(_step("release-check", "check-release", release, release_pf,
                       retries=0, timeout_s=2 * 3600))
    if r.score:
        caches = _caches(r.teacher)
        if caches:
            kl = [a for c in caches for a in ("--cache", c)] + [
                "--rung", f"{r.name}={pin}", "--per-pos-dir", f"{r.work}/kl-per-pos",
                "--out", f"{r.work}/kl.json", "--lazy-over-gb", 16]
            warns.append("The score is this build's KL to the teacher, not a paired comparison "
                         "against a published build.")
            steps.append(_step("score", "kl-ladder", kl, {"append": ["--preflight"]},
                               retries=1, timeout_s=12 * 3600,
                               expect={"stdout_regex": "KL", "files": [f"{r.work}/kl.json"]}))
        else:
            warns.append("No KL caches are registered for this teacher, so the build is not scored.")
    return steps, warns


def queue_text(q: dict) -> str:
    """A compiled queue as its file's text; every writer of one uses it."""
    return json.dumps(q, indent=1) + "\n"


def compile_recipe(r: dict) -> tuple[dict, list[str]]:
    """-> (queue file dict, warnings). Raises RecipeError, and only that:
    a recipe or profile missing a key it needs is refused, not a crash."""
    try:
        return _compile(r)
    except KeyError as e:
        raise RecipeError(f"the recipe or the teacher's profile is malformed "
                          f"(missing key {e})") from None


def _compile(raw: dict) -> tuple[dict, list[str]]:
    r = Recipe.from_json(raw)
    prof = load_profile(r.family, r.teacher)
    ident = prof["identity"]
    src = source_path(teacher_profiles.teacher_source(ident) or ident["path"])
    kind = prof["arch"].get("kind")
    if (r.family in MOE) != (kind == "moe"):
        raise RecipeError(f"{r.teacher} is {kind}, but {r.family} builds are compiled as "
                          f"{'moe' if r.family in MOE else 'dense'}")
    by = prof["bytes"]["by_class_gib"]
    bad = {k: v for k, v in by.items() if isinstance(v, bool) or not isinstance(v, (int, float))}
    if bad:
        raise RecipeError(f"the teacher's profile is malformed: bytes.by_class_gib {bad} "
                          "is not a number of GiB")
    in_base = _fam(r.family).get("tower") == "base"      # kept whatever the recipe says
    vis = by.get("vision", 0.0) if r.vision or in_base else 0.0
    if r.mode == "affine":
        steps, art_gib, scratch_gib, pf_gib, warns = _affine_steps(r, src, by, vis)
    else:
        chain = _moe_steps if r.family in MOE else _dense_steps
        steps, art_gib, scratch_gib, pf_gib, warns = chain(r, prof, src, by, vis)
    if r.skipzero:
        vq, _ = _layers(prof)
        geo = {i: (r.tail[1:] if r.tail and i >= r.tail[0] else r.geo) for i in vq}
        saved = skipzero_gib(prof, _skipzero_rows(r), geo)
        art_gib -= saved
        scratch_gib += art_gib      # at most: sz-pack rewrites every shard that holds a dead row
    head = mtp_head(r.family, r.teacher, prof, r.mtp_bits) if r.mtp else None
    if head:
        art_gib += head.gib
        # the bf16 graft, the sidecar, and a trial one where the probe chooses
        scratch_gib += head.graft_gib + head.gib * (1 if head.wiring is not None else 2)
    # the resident bar of THIS box (0.90 x its RAM, read from the machine),
    # the rule preflight-ram enforces at load time. A build is gated where
    # its preflight can be gated too: the chain's preflight artifact is its
    # base (Flash-Next: ~69 GiB built, ~94 GiB struct-base preflight).
    pf_gib += head.gib if head else 0.0
    oversized = max(art_gib, pf_gib) > preflight_ram.bar_gib()
    tail, tail_warns = _tail_steps(r, src, by, head, oversized=oversized)
    return {"name": r.name, "commit": "HEAD", "recipe": r.raw,
            # estimates for the confirmation, not measurements: the artifact
            # from the profile's exact per-geometry bytes, scratch = every
            # intermediate the chain leaves (struct base, unpacked build)
            "estimate": {"artifact_gib": round(art_gib, 2), "scratch_gib": round(scratch_gib, 1),
                         "preflight_gib": round(pf_gib, 2)},
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
    if a.out:
        write_text_atomic(a.out, queue_text(q))
        print(f"wrote {a.out}; run it with `vqlab queue run {a.out} --preflight`")
    else:
        print(queue_text(q), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
