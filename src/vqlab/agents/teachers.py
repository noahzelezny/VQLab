"""The teachers a build can start from, for the GUI's Recipe and Library.

    /api/teachers           profiled teachers on this machine + originals
                            that only need profiling
    POST /api/teachers/profile {path}   profile one (headers only, seconds)

Everything here is read from what the lab already keeps, never estimated:
the family profile (`vqlab family-profile`: exact bytes per class, the
legal (d, K) table with each geometry's exact size), the fit store (which
geometries are already fitted), the teacher's registered KL caches, and the
model scan (`vqlab lab scan`) for originals nobody has profiled yet. A
teacher's sizes in GiB come from the profile; a build's size is the carried
bytes plus the chosen geometry's exact stack, which is arithmetic, not a fit
to past rungs.
"""
from __future__ import annotations

import os
import subprocess
import threading

from vqlab import config
from vqlab.agents import model_scan
from vqlab.util.errors import UserError
from vqlab.core.source import fit_store
from vqlab.family import registry
from vqlab.plan import recipe_queue
from vqlab.queue import files as qf

_PROFILING: set[str] = set()
_PROFILE_ERRORS: dict[str, str] = {}     # realpath -> why its last profile failed
_LOCK = threading.Lock()


def _fits() -> dict:
    """{(family, teacher): {"dD-KK": modules fitted}} from every fit store."""
    out: dict = {}
    for root in fit_store.roots():
        for r in fit_store.read_index(root).values():
            g = out.setdefault((r["family"], r["teacher"]), {})
            key = recipe_queue.geo_key(r["d"], r["K"])
            g[key] = g.get(key, 0) + 1
    return out


def _geometries(prof: dict) -> list[dict]:
    """Every (d, K) legal on ALL projections, with the expert stack's exact
    GiB and whether a fit can be compiled at it (plan/recipe_queue.py)."""
    return [{"geo": recipe_queue.geo_key(g.d, g.K), "d": g.d, "K": g.K, "bpw": g.bpw,
             "gib": round(g.gib, 3), "fit": recipe_queue.has_relerr(g.d, g.K)}
            for g in recipe_queue.legal_geometries(prof)]


def _skipzero(family: str, teacher: str, prof: dict) -> dict | None:
    """What skip-zero saves on this teacher, per VQ layer and geometry, from
    its measured dead rows (plan/recipe_queue.skipzero_gib, the compiler's
    own figure), with where the counts came from. None where no dead rows
    were measured or the compiler would refuse skipzero."""
    if family not in recipe_queue.MOE:
        return None
    sz = registry.teacher_skipzero(family, teacher)
    if not sz or not sz.get("dead_rows"):
        return None
    gib = {}
    for g in recipe_queue.legal_geometries(prof):
        try:
            per = {str(i): recipe_queue.skipzero_gib(prof, sz, {int(i): (g.d, g.K)})
                   for i in sz["dead_rows"]}
        except (recipe_queue.RecipeError, KeyError, TypeError, ValueError):
            continue
        gib[recipe_queue.geo_key(g.d, g.K)] = {i: round(v, 5) for i, v in per.items() if v}
    if not gib:
        return None
    return {"gib": gib, "source": sz.get("source"), "findings": sz.get("findings") or [],
            "checks": sz.get("checks") or []}


def _scanned() -> list[dict]:
    """The model scan's rows; an unreadable index reads as no models."""
    try:
        return model_scan.load()
    except OSError:
        return []


def _local(path: str) -> str | None:
    p = config.expand(path or "")
    return os.path.realpath(p) if p and not p.startswith("hf://") and os.path.isdir(p) else None


def teachers() -> dict:
    fits = _fits()
    caches = {c["teacher"] for c in registry.teacher_caches()}
    out, have, names = [], set(), set()
    for (family, teacher), prof in sorted(registry.iter_profiles().items()):
        if prof.get("unknown_family"):
            continue
        arch, by = prof.get("arch") or {}, (prof.get("bytes") or {})
        local = _local((prof.get("identity") or {}).get("path"))
        if local:
            have.add(local)
            names.add(teacher)
        out.append({
            "id": f"{family}/{teacher}", "family": family, "teacher": teacher,
            "name": (prof.get("identity") or {}).get("hf_repo") or teacher.replace("--", "/", 1),
            "kind": arch.get("kind"), "model_type": arch.get("model_type"),
            "layers": arch.get("layers"), "vq_layers": len(arch.get("vq_layers") or []),
            "first_vq": min(arch.get("vq_layers") or [0]), "experts": arch.get("experts"),
            "topk": arch.get("experts_per_token"), "modules": (prof.get("modules") or {}).get("count"),
            "gib": by.get("by_class_gib") or {}, "gib_per_bit": by.get("vq_target_gib_per_bit"),
            "geometries": _geometries(prof), "fits": fits.get((family, teacher), {}),
            "on_disk": local is not None, "buildable": family in recipe_queue.MOE | recipe_queue.DENSE,
            "scored": teacher in caches, "profiled": prof.get("created"),
            "skipzero": _skipzero(family, teacher, prof)})
    found = []
    for r in _scanned():
        if r.get("use") != "original" or not r.get("path") or not r.get("family"):
            continue                 # quantized, or not a model family vqlab builds
        real = os.path.realpath(r["path"])
        # another copy of a teacher already profiled here is that teacher
        if real in have or (r.get("name") or "").replace("/", "--") in names:
            continue
        have.add(real)
        found.append({"name": r.get("name"), "path": r["path"], "family": r.get("family"),
                      "model_type": r.get("model_type"), "bytes": r.get("bytes"),
                      "profiling": real in _PROFILING, "profile_error": _PROFILE_ERRORS.get(real)})
    return {"teachers": out, "unprofiled": found}


def profile(body: dict) -> dict:
    """Profile one scanned original in the background: `vqlab family-profile
    --teacher PATH` reads headers only and writes the user's families folder.
    Only a path the model scan found is accepted."""
    path = str(body.get("path") or "")
    known = {os.path.realpath(r["path"]) for r in _scanned() if r.get("path")}
    real = os.path.realpath(path)
    if not path or real not in known:
        raise UserError("not a model the scan found; rescan in Storage first")
    with _LOCK:
        if real in _PROFILING:
            return {"ok": True, "running": True}
        _PROFILING.add(real)
        _PROFILE_ERRORS.pop(real, None)

    def go():
        err = None
        try:
            r = qf.run_cli("family-profile", "--teacher", path, timeout=1800)
            if r.returncode:
                tail = qf.output_tail(r, 3).replace("\n", " | ")
                err = f"family-profile exited {r.returncode}" + (f": {tail}" if tail else "")
        except subprocess.TimeoutExpired:
            err = "family-profile timed out after 30 min"
        except OSError as exc:
            err = f"could not run family-profile: {exc}"
        finally:                         # whatever happened, the row stops saying "profiling"
            with _LOCK:
                _PROFILING.discard(real)
                if err:
                    _PROFILE_ERRORS[real] = err
    threading.Thread(target=go, daemon=True).start()
    return {"ok": True, "running": True}
