#!/usr/bin/env python3
"""The NUMERICS BUILD a number was computed under, and the check that two match.

F194: scoring (mlx 0.32.0.dev + a grafted mlx-lm 0.31.9) and serving (mlx
0.31.2) disagree by up to 1.0 logit on DeepSeek while the architecture code is
bitwise identical. A KL number is only meaningful for the build that produced
it, so every teacher cache, every scored record and every per-position array
carries this stamp:

    mlx             mlx's version (the kernels)
    mlx_lm          mlx-lm's version (the runtime that builds the graph)
    arch_file       the runtime library's architecture module for the
                    model_type (mlx_lm.models.<mt> or mlx_vlm.models.<mt>), as
                    RESOLVED in this interpreter -- a vendored module already
                    in sys.modules (Knurlogic's register) wins, as it does at
                    load time
    arch_sha256     its sha256
    scorer_variant  the family plugin's scorer variant (None without one)
    arch_files      {module: sha256} for the arch file AND every module of the
                    runtime's models package it imports, transitively (static
                    AST walk; each module resolved the way the import resolves
                    it here, so a vendored sibling wins). A MoE subclass file
                    imports its base file, and the base holds the attention /
                    GatedDeltaNet code; arch_sha256 alone never saw it
    arch_tree_sha256  sha256 over arch_files. check_cache / pairing compare it
                    in place of arch_sha256 when BOTH stamps carry it; an older
                    stamp without it is compared on arch_sha256, with a note
                    that its imported files are not covered

The ARCH file, not the artifact's bundled model.py, is what is compared: a
teacher and its VQ student load different model.py files by design (the
bundle splices the VQ runtime around `mlx_lm.models.<mt>`), so their bundles
never match while the architecture they both call must. The file the model
class actually came from is recorded beside it (`loaded_file`/`loaded_sha256`)
and is not compared.

    python -m vqlab.family.version_stamp --model <dir>    # this interpreter's build, JSON

Reads files and package metadata only; never touches the GPU.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import pathlib
import sys
from vqlab.core.artifacts.stamp_fields import (  # noqa: F401  (re-exported)
    COMPARED, describe, diff, tree_compared)
from vqlab.util.hashing import sha256_file

_UNSET = object()


def _sha(path) -> str | None:
    return sha256_file(path, missing_ok=True)


def _version(dist: str, module: str) -> str | None:
    # The imported module's own __version__ first: a grafted install can keep
    # the pip metadata of the build it replaced.
    mod = sys.modules.get(module)
    v = getattr(mod, "__version__", None) if mod else None
    if v:
        return str(v)
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def _knurlogic() -> str | None:
    """knurlogic's version + commit (its architecture files are the ones
    loaded). Recorded, not compared: arch_sha256 already compares the bytes
    that matter, and a knurlogic commit that leaves them alone changes no
    number."""
    try:
        d = importlib.metadata.distribution("knurlogic")
    except importlib.metadata.PackageNotFoundError:
        return None
    commit = None
    try:
        commit = (json.loads(d.read_text("direct_url.json") or "{}")
                  .get("vcs_info", {}).get("commit_id"))
    except ValueError:
        pass
    return d.version + (f"@{commit[:9]}" if commit else "")


def versions() -> dict:
    return {"mlx": _version("mlx", "mlx.core") or _version("mlx", "mlx"),
            "mlx_lm": _version("mlx-lm", "mlx_lm"),
            "knurlogic": _knurlogic()}


def model_type_of(model_dir) -> str | None:
    cfg = json.loads((pathlib.Path(model_dir) / "config.json").read_text())
    return cfg.get("model_type") or cfg.get("text_config", {}).get("model_type")


def _runtime_for(family: str | None) -> str:
    if not family:
        return "mlx_lm"
    try:
        from vqlab.family import model_loader as runtime_load
        return runtime_load.runtime_for(family)
    except Exception:
        return "mlx_lm"


def arch_file(model_type: str | None, runtime: str = "mlx_lm", *,
              remap: bool = True) -> str | None:
    """Path of the architecture module this interpreter resolves for model_type.
    `remap=False` resolves a module NAME (an imported sibling), not a model_type."""
    if not model_type:
        return None
    mt = model_type
    if runtime == "mlx_lm" and remap:
        try:
            from mlx_lm.utils import MODEL_REMAPPING
            mt = MODEL_REMAPPING.get(mt, mt)
        except Exception:
            pass
    name = f"{runtime}.models.{mt}"
    mod = sys.modules.get(name)
    if mod is not None:
        return getattr(mod, "__file__", None)
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    return spec.origin if spec else None


def _resolve_module(dotted: str, runtime: str) -> str | None:
    """File serving `dotted`, resolved as its import would be: sys.modules
    first; a top-level module of <runtime>.models through the import system
    (so a vendored sibling wins, as at load time); anything deeper on disk
    inside its parent package's dir. Never imports outside <runtime>.models."""
    mod = sys.modules.get(dotted)
    if mod is not None:
        return getattr(mod, "__file__", None)
    prefix = f"{runtime}.models."
    if dotted.startswith(prefix) and "." not in dotted[len(prefix):]:
        return arch_file(dotted[len(prefix):], runtime, remap=False)
    if "." not in dotted:
        return None
    parent, last = dotted.rsplit(".", 1)
    pf = _resolve_module(parent, runtime)
    if not pf or pathlib.Path(pf).name != "__init__.py":
        return None
    d = pathlib.Path(pf).parent
    for cand in (d / f"{last}.py", d / last / "__init__.py"):
        if cand.is_file():
            return str(cand)
    return None


def _imported_modules(path: str, dotted: str) -> set:
    """Candidate module names the file imports, relative imports made absolute
    against `dotted` (`from .x import Y` yields x and x.Y; a name that is not a
    module resolves to no file later and drops out)."""
    try:
        tree = ast.parse(pathlib.Path(path).read_bytes())
    except (OSError, SyntaxError, ValueError):
        return set()
    package = dotted if pathlib.Path(path).name == "__init__.py" else dotted.rsplit(".", 1)[0]
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                if node.level - 1 >= len(parts):
                    continue
                base = ".".join(parts[:len(parts) - (node.level - 1)])
                base = f"{base}.{node.module}" if node.module else base
            else:
                base = node.module or ""
            if node.module:      # `from . import x` names x, not the package
                out.add(base)
            out |={f"{base}.{a.name}" for a in node.names if a.name != "*"}
    out.discard(dotted)
    return out


def arch_files(model_type: str | None, runtime: str = "mlx_lm") -> dict:
    """{module: sha256} for the model_type's arch file and every arch module it
    imports, transitively: modules of <runtime>.models (keyed relative to it),
    and modules under the arch file's own directory reached by an absolute
    import (a vendored package importing its private copy; keyed by full
    name). Anything else (mlx, numpy, transformers) is outside the stamp."""
    af = arch_file(model_type, runtime)
    if not af:
        return {}
    p = pathlib.Path(af)
    pkg = p.name == "__init__.py"
    home = (p.parent.parent if pkg else p.parent).resolve()
    prefix = f"{runtime}.models."
    root = prefix + (p.parent.name if pkg else p.stem)
    files, todo, seen = {}, [(root, af)], {root}
    while todo:
        dotted, path = todo.pop()
        key = dotted[len(prefix):] if dotted.startswith(prefix) else dotted
        files[key] = _sha(path)
        for m in sorted(_imported_modules(path, dotted)):
            if m in seen:
                continue
            seen.add(m)
            f = _resolve_module(m, runtime)
            if not f or not f.endswith(".py"):
                continue
            if not m.startswith(prefix) and home not in pathlib.Path(f).resolve().parents:
                continue
            todo.append((m, f))
    return dict(sorted(files.items()))


def tree_sha256(files: dict) -> str | None:
    """sha256 over {module: sha256} as canonical JSON (sorted keys, compact)."""
    if not files:
        return None
    blob = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def scorer_variant(model_type: str | None):
    """family.registry.scorer_variant (moved there 2026-10-10; this name is
    kept for one release). Imported at call time, as before the move, so
    reading a stamp never loads the family plugins."""
    from vqlab.family.registry import scorer_variant as _variant
    return _variant(model_type)


def build(model_type: str | None = None, *, family: str | None = None,
          model=None, model_dir=None, variant=_UNSET) -> dict:
    """This interpreter's numerics build for a model_type.

    `variant` defaults to the family plugin's current scorer variant; a path
    that does not apply the variant (kl_damage's direct forward) passes None.
    `model`, when loaded, adds the file its class actually came from."""
    if model_type is None and model_dir is not None:
        model_type = model_type_of(model_dir)
    runtime = _runtime_for(family)
    af = arch_file(model_type, runtime)
    out = {**versions(), "model_type": model_type, "runtime": runtime,
           "arch_file": af, "arch_sha256": _sha(af) if af else None,
           "scorer_variant": scorer_variant(model_type) if variant is _UNSET else variant}
    files = arch_files(model_type, runtime) if af else {}
    out["arch_files"], out["arch_tree_sha256"] = files, tree_sha256(files)
    if model is not None:
        mod = sys.modules.get(type(model).__module__)
        lf = getattr(mod, "__file__", None)
        if lf:
            out["loaded_file"], out["loaded_sha256"] = lf, _sha(lf)
    return out


ARCH_EQUIVALENCE = pathlib.Path(__file__).with_name("arch_equivalence.json")


def arch_equivalence(a: dict | None, b: dict | None, bad: list, path=None):
    """The arch_equivalence.json entry that excuses `bad`, else None.

    Only when the ONLY differing compared field is the arch hash (arch_sha256,
    or arch_tree_sha256 when both stamps carry it), both stamps name the same
    model_type, and that (sha, sha) pair is listed for it under that field
    (an entry's `field`, default arch_sha256) -- content-addressed, either
    direction. Anything else stays a mismatch."""
    if len(bad) != 1 or bad[0][0] not in ("arch_sha256", "arch_tree_sha256"):
        return None
    field, x, y = bad[0]
    mt = (a or {}).get("model_type")
    if not x or not y or not mt or mt != (b or {}).get("model_type"):
        return None
    try:
        reg = json.loads(pathlib.Path(path or ARCH_EQUIVALENCE).read_text())
    except (OSError, ValueError):
        return None
    for e in reg.get("entries", []):
        if (e.get("model_type") == mt and e.get("field", "arch_sha256") == field
                and {e.get("old_sha256"), e.get("new_sha256")} == {x, y}):
            return e
    return None


def _note_equivalence(e: dict, what: str) -> None:
    print(f"NOTE: the {what}'s arch file differs from this run's only by an entry in "
          f"{ARCH_EQUIVALENCE.name} ({e['model_type']} {e.get('field', 'arch_sha256')}: "
          f"{e['old_sha256'][:12]} == "
          f"{e['new_sha256'][:12]}, {e.get('date')}): bit-identical output, accepted.",
          flush=True)


def _note_tree_coverage(a: dict | None, b: dict | None, what: str) -> None:
    """An older stamp has no arch_tree_sha256: say what was not compared."""
    if tree_compared(a, b):
        return
    print(f"NOTE: the {what} or this run predates arch_tree_sha256; compared on "
          f"arch_sha256 (the model_type's own arch file) only. Whether the arch "
          f"modules it imports (e.g. a MoE subclass's base file) match is UNKNOWN "
          f"for this {what}.", flush=True)


def check_cache(meta: dict, run: dict, allow: bool = False, what: str = "teacher cache") -> dict:
    """Refuse when a cache's build differs from this run's; return the record.

    A cache written before the stamp existed (no `numerics`) is WARNED about,
    not refused: only its top-level scorer_variant (stamped since F195) can be
    checked. `allow` turns a refusal into a recorded override."""
    cached = meta.get("numerics")
    if cached is None:
        msg = (f"WARNING: the {what} carries NO numerics build stamp (written before "
               f"2026-10-03). mlx, mlx-lm and the arch file cannot be checked; this run "
               f"is {describe(run)}. Rebuild the cache to make the KL attributable.")
        print(msg, flush=True)
        print(msg, file=sys.stderr, flush=True)
        bad = [("scorer_variant", meta.get("scorer_variant"), run.get("scorer_variant"))]
        bad = [x for x in bad if x[1] != x[2]]
        status = "unstamped-cache"
    else:
        bad = diff(cached, run)
        _note_tree_coverage(cached, run, what)
        status = "match"
        eq = arch_equivalence(cached, run, bad)
        if eq is not None:
            _note_equivalence(eq, what)
            return {"status": "arch-equivalent", "mismatches": [list(x) for x in bad],
                    "equivalence": {k: eq.get(k) for k in
                                    ("model_type", "old_sha256", "new_sha256", "date")}}
    if bad:
        lines = "; ".join(f"{k}: cache {c!r} vs run {r!r}" for k, c, r in bad)
        if not allow:
            raise SystemExit(
                f"FAIL: the {what} was built with different numerics than this run ({lines}). "
                "Teacher and student must be computed by the same build, or the KL measures "
                "the build, not the quantization. Rebuild the cache in this environment, or "
                "pass --allow-build-mismatch (recorded in the result).")
        print(f"WARNING: build mismatch ALLOWED by --allow-build-mismatch: {lines}", flush=True)
        status = "mismatch-allowed"
    return {"status": status, "mismatches": [list(x) for x in bad]}


# ------------------------------------------------------------ cache identity
def cache_identity(cache_dir) -> dict:
    """What makes two per-position arrays pairable: the SAME positions
    (tokens.safetensors) against the SAME teacher (its stored logprobs, head-
    hashed: a full-vocab cache is 12 GB) and the build the cache was made by."""
    cd = pathlib.Path(cache_dir)
    out = {"dir": str(cd), "tokens_sha256": _sha(cd / "tokens.safetensors")}
    for name in ("teacher_full.safetensors", "teacher_topk.safetensors"):
        p = cd / name
        if p.exists():
            h = hashlib.sha256()
            with open(p, "rb") as f:
                h.update(f.read(1 << 20))
            out["teacher_file"] = name
            out["teacher_head_sha256"] = h.hexdigest()
            out["teacher_bytes"] = p.stat().st_size
            break
    try:
        out["numerics"] = json.loads((cd / "meta.json").read_text()).get("numerics")
    except (OSError, ValueError):
        out["numerics"] = None
    return out


def pairing_problems(a: dict | None, b: dict | None) -> list:
    """Why two per-position sidecar records can NOT be paired ([] = pairable)."""
    if not a or not b:
        return ["a per-position array has no .json sidecar"]
    probs = []
    ca, cb = a.get("kl_cache"), b.get("kl_cache")
    if not ca or not cb:
        probs.append("a sidecar predates the cache identity stamp (kl_cache)")
    else:
        for k in ("tokens_sha256", "teacher_head_sha256", "teacher_bytes"):
            if ca.get(k) != cb.get(k):
                probs.append(f"different cache: {k} {ca.get(k)!r} vs {cb.get(k)!r}")
    na, nb = a.get("numerics"), b.get("numerics")
    if not na or not nb:
        probs.append("a sidecar carries no numerics build stamp")
    else:
        bad = diff(na, nb)
        _note_tree_coverage(na, nb, "per-position array")
        eq = arch_equivalence(na, nb, bad)
        if eq is not None:
            _note_equivalence(eq, "per-position array")
        else:
            probs += [f"different build: {k} {x!r} vs {y!r}" for k, x, y in bad]
    if a.get("kl_positions") != b.get("kl_positions"):
        probs.append(f"different position count {a.get('kl_positions')} vs {b.get('kl_positions')}")
    return probs


def stamp(model_type: str | None = None, *, family: str | None = None,
          model_dir=None) -> dict:
    """build() for a model_type, or the one a model dir's config names; the
    family defaults to the one the family registry maps that model_type to."""
    mt = model_type or (model_type_of(model_dir) if model_dir else None)
    if family is None and mt:
        try:
            from vqlab.family import model_loader as runtime_load
            family = runtime_load.family_for_model_type(mt)
        except Exception:
            family = None
    return build(mt, family=family)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", help="model dir (its config.json names the model_type)")
    ap.add_argument("--model-type")
    ap.add_argument("--family")
    a = ap.parse_args(argv)
    print(json.dumps(stamp(a.model_type, family=a.family, model_dir=a.model)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
