"""Read-only storage operations: where_is, list_artifacts, artifact_config,
disk_free. None of them writes anything."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from vqlab.ops.errors import LabError
from vqlab.ops.roots import _dir_bytes, _under, _walk, lab_roots


def _teacher_cache_hits(needle: str, kind: str, known: set) -> List[Dict[str, Any]]:
    """Registered teacher caches match on their TEACHER or corpus too, not
    only their directory name: "397B" finds q397_prose_12k."""
    out: List[Dict[str, Any]] = []
    try:
        from vqlab.family.registry import teacher_caches
        for c in teacher_caches():
            if needle in (c["teacher"] + " " + c.get("corpus", "") + " " + c["path"]).lower() \
                    and c["path"] not in known and kind in ("any", "dir"):
                out.append({"path": c["path"], "kind": "teacher_cache",
                            "teacher": c["teacher"], "corpus": c.get("corpus"),
                            "tokens": c.get("tokens"), "full_vocab": c.get("full_vocab"),
                            "exists": Path(c["path"]).is_dir()})
    except Exception:  # the registry is a bonus, never a failure
        pass
    return out


def where_is(name: str, kind: str = "any", max_depth: int = 3) -> Dict[str, Any]:
    """Find a thing by (sub)name across the lab roots. Deterministic."""
    if not name or len(name) < 3:
        raise LabError("INVALID_ARG", "name must be at least 3 characters")
    needle = name.lower()
    roots = lab_roots()
    hits: List[Dict[str, Any]] = []
    for root, e, _ in _walk(roots, max_depth):
        if needle not in e.name.lower():
            continue
        is_dir = e.is_dir(follow_symlinks=False)
        if kind == "dir" and not is_dir or kind == "file" and is_dir:
            continue
        rec: Dict[str, Any] = {"path": e.path, "kind": "dir" if is_dir else "file",
                               "root": str(root)}
        if is_dir:
            p = Path(e.path)
            rec["has_config"] = (p / "config.json").is_file()
            rec["safetensors_gib"] = round(_dir_bytes(p) / 2**30, 3)
        else:
            try:
                rec["bytes"] = e.stat(follow_symlinks=False).st_size
            except OSError:
                pass
        hits.append(rec)
        if len(hits) >= 50:
            break
    hits.sort(key=lambda h: h["path"])
    hits += _teacher_cache_hits(needle, kind, {h["path"] for h in hits})
    if not hits:
        return {"found": False, "status": "NOT_FOUND", "name": name,
                "roots_searched": [str(r) for r in roots],
                "roots_missing": [str(r) for r in roots if not r.is_dir()],
                "max_depth": max_depth,
                "note": "Searched every lab root to this depth. If you expected it, "
                        "the name is different, not the location."}
    return {"found": True, "count": len(hits), "matches": hits,
            "roots_searched": [str(r) for r in roots]}


def list_artifacts(root: str = "", max_depth: int = 3) -> Dict[str, Any]:
    """Directories with a config.json under the lab roots (or one root)."""
    roots = [Path(root)] if root else lab_roots()
    if root and not _under(Path(root), lab_roots()):
        raise LabError("OUTSIDE_ROOTS", f"{root} is not under a lab root",
                       roots=[str(r) for r in lab_roots()])
    out = []
    for r, e, _ in _walk(roots, max_depth):
        if e.is_dir(follow_symlinks=False) and (Path(e.path) / "config.json").is_file():
            p = Path(e.path)
            out.append({"path": e.path, "safetensors_gib": round(_dir_bytes(p) / 2**30, 3),
                        "has_model_py": (p / "model.py").is_file(),
                        "has_readme": (p / "README.md").is_file()})
    out.sort(key=lambda a: a["path"])
    return {"count": len(out), "artifacts": out, "roots": [str(r) for r in roots]}


def artifact_config(path: str, keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """The shipped config.json -- the record of what actually shipped."""
    p = Path(path)
    if not _under(p, lab_roots()):
        raise LabError("OUTSIDE_ROOTS", f"{path} is not under a lab root")
    cfg = p / "config.json" if p.is_dir() else p
    if not cfg.is_file():
        raise LabError("NOT_FOUND", f"no config.json at {p}")
    data = json.loads(cfg.read_text())
    if keys:
        data = {k: data.get(k) for k in keys}
    text = json.dumps(data, indent=1)
    if len(text) > 20000:
        text = text[:20000] + "\n… (truncated; pass keys=[...] to narrow)"
    return {"path": str(cfg), "config": text}


def disk_free() -> Dict[str, Any]:
    """Free space on every configured storage root (writers run out mid-run)."""
    seen, out = set(), []
    for r in lab_roots():
        try:
            u = shutil.disk_usage(r)
        except OSError:
            continue
        key = (u.total, u.free)
        out.append({"root": str(r), "free_gib": round(u.free / 2**30, 1),
                    "total_gib": round(u.total / 2**30, 1), "same_volume_as_previous": key in seen})
        seen.add(key)
    return {"roots": out}
