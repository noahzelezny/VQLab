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

import json
import os
import pathlib
import subprocess
import threading

from vqlab import config
from vqlab.family import registry
from vqlab.plan import recipe_queue
from vqlab.queue import files as qf

_PROFILING: set[str] = set()
_LOCK = threading.Lock()


def _profiles():
    """(family, teacher, profile) for every profile; the checkout's first,
    the user's overriding (as the registry reads them)."""
    seen = {}
    for root in registry._family_dirs():
        for p in sorted(pathlib.Path(root).glob("*/teachers/*/profile.json")):
            try:
                seen[(p.parent.parent.parent.name, p.parent.name)] = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
    return seen


def _fits() -> dict:
    """{(family, teacher): {"dD-KK": modules fitted}} from every fit store."""
    from vqlab.core.source import fit_store
    out: dict = {}
    for root in fit_store.roots():
        for r in fit_store.read_index(root).values():
            g = out.setdefault((r["family"], r["teacher"]), {})
            key = f"d{r['d']}-K{r['K']}"
            g[key] = g.get(key, 0) + 1
    return out


def _geometries(prof: dict) -> list[dict]:
    """Every (d, K) legal on ALL projections, with the expert stack's exact
    GiB summed over them and whether a fit can be compiled at it."""
    tables = prof.get("geometries") or {}
    rows = {}
    for proj, entries in tables.items():
        for e in entries:
            r = rows.setdefault((e["d"], e["K"]), {"d": e["d"], "K": e["K"], "gib": 0.0, "n": 0,
                                                  "bpw": e.get("bits_per_weight"), "legal": True})
            r["gib"] += e.get("stack_gib", 0.0)
            r["n"] += 1
            r["legal"] &= bool(e.get("legal")) and e.get("fused_kernel_ok", True) is not False
    out = []
    for (d, k), r in sorted(rows.items()):
        if r["n"] != len(tables) or not r["legal"]:
            continue
        try:
            recipe_queue.relerr_abort(d, k)
            fit = True
        except recipe_queue.RecipeError:
            fit = False
        out.append({"geo": f"d{d}-K{k}", "d": d, "K": k, "bpw": r["bpw"],
                    "gib": round(r["gib"], 3), "fit": fit})
    return out


def _local(path: str) -> str | None:
    p = config.expand(path or "")
    return os.path.realpath(p) if p and not p.startswith("hf://") and os.path.isdir(p) else None


def teachers() -> dict:
    fits = _fits()
    caches = {c["teacher"] for c in registry.teacher_caches()}
    out, have, names = [], set(), set()
    for (family, teacher), prof in sorted(_profiles().items()):
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
            "scored": teacher in caches, "profiled": prof.get("created")})
    found = []
    try:
        from vqlab.agents import model_scan
        rows = model_scan.load()
    except (OSError, ValueError):
        rows = []
    for r in rows:
        if r.get("use") != "original" or not r.get("path") or not r.get("family"):
            continue                 # quantized, or not a model family vqlab builds
        real = os.path.realpath(r["path"])
        # another copy of a teacher already profiled here is that teacher
        if real in have or (r.get("name") or "").replace("/", "--") in names:
            continue
        have.add(real)
        found.append({"name": r.get("name"), "path": r["path"], "family": r.get("family"),
                      "model_type": r.get("model_type"), "bytes": r.get("bytes"),
                      "profiling": real in _PROFILING})
    return {"teachers": out, "unprofiled": found}


def profile(body: dict) -> dict:
    """Profile one scanned original in the background: `vqlab family-profile
    --teacher PATH` reads headers only and writes the user's families folder.
    Only a path the model scan found is accepted."""
    path = str(body.get("path") or "")
    from vqlab.agents import model_scan
    known = {os.path.realpath(r["path"]) for r in model_scan.load() if r.get("path")}
    real = os.path.realpath(path)
    if not path or real not in known:
        return {"ok": False, "error": "not a model the scan found; rescan in Storage first"}
    with _LOCK:
        if real in _PROFILING:
            return {"ok": True, "running": True}
        _PROFILING.add(real)

    def go():
        try:
            pre, env, cwd = qf.cli_launch()
            subprocess.run([*pre, "family-profile", "--teacher", path], env=env, cwd=cwd,
                           capture_output=True, text=True, timeout=1800)
        finally:
            with _LOCK:
                _PROFILING.discard(real)
    threading.Thread(target=go, daemon=True).start()
    return {"ok": True, "running": True}
