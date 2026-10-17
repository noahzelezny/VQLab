"""The package dependency rule (lab/docs PACKAGE-LAYOUT-PLAN §2), enforced.

A package may import only from LOWER layers; peers in one layer may not
import each other. Every import counts, including ones inside function
bodies (a lazy import is still an edge). The same walk also refuses new
`sys.path` edits, new bare imports of vqlab module stems (`import vq_pack`
instead of `from vqlab.runtime import vq_pack`), and new `parents[N]` (N>=2)
repo-relative path arithmetic.

Today's tree breaks these rules in known places; those are listed in
tests/fixtures/layering_allowlist.json. The test fails on any violation NOT
in the list, and on any listed violation that no longer exists, so the list
only ever shrinks: when a refactor step removes violations, regenerate it

    PYTHONPATH=src python tests/test_layering.py --write

and review the diff (it must only lose lines).
"""
from __future__ import annotations

import ast
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "vqlab"
ALLOWLIST = ROOT / "tests" / "fixtures" / "layering_allowlist.json"

# package -> layer. Higher may import lower; equal = peers (forbidden unless
# the same package). `commands` is pure data (strings only), so it sits at
# L0 and anything may read it.
LAYER = {
    "util": 0, "config": 0, "commands": 0,
    "runtime": 0.5,
    "core": 1,
    "records": 2,
    "family": 3,
    "mtp": 4,
    "plan": 5, "fit": 5, "skipzero": 5,
    "build": 6,
    "bundle": 7,
    "gate": 8, "score": 8,
    "bench": 9,
    "ship": 10,
    "queue": 11,
    "ops": 11.5,            # lab operations on this box: runs, storage, findings
    "api": 11.75,           # the stable surface; consumers (agents, cli) sit above
    "agents": 12, "selftest": 12,
    "cli": 13,
}
# Infrastructure every layer may name: the package root (public MTP API,
# use_knurlogic_arch) and the import-alias table. Both shrink in later steps.
EXEMPT = {"", "_layout"}
# Peer edges the plan allows explicitly.
ALLOWED_PEERS = {("util", "config")}


def _dotted_aliases():
    """{alias: canonical dotted name} -- the names _layout resolves."""
    sys.path.insert(0, str(ROOT / "src"))
    from vqlab import _layout
    return _layout.PERMANENT


def _pkg_of(dotted: str) -> str | None:
    """'vqlab.fit.geo_build' -> 'fit'; 'vqlab' -> ''; non-vqlab -> None."""
    if dotted == "vqlab":
        return ""
    if not dotted.startswith("vqlab."):
        return None
    return dotted.split(".")[1]


def _module_name(path: pathlib.Path) -> str:
    rel = path.relative_to(PKG.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _imported(node, modname, is_pkg):
    """Dotted names one import node pulls in (a `from vqlab import x` names
    the submodule vqlab.x when x is one)."""
    if isinstance(node, ast.Import):
        return [a.name for a in node.names]
    base = node.module or ""
    if node.level:
        parts = modname.split(".")
        if not is_pkg:
            parts = parts[:-1]
        parts = parts[: len(parts) - (node.level - 1)]
        base = ".".join(parts + ([base] if base else []))
    out = []
    for a in node.names:
        cand = f"{base}.{a.name}" if base else a.name
        sub = PKG.parent / pathlib.Path(*cand.split("."))
        if cand.startswith("vqlab.") and (sub.with_suffix(".py").exists() or sub.is_dir()):
            out.append(cand)
        else:
            out.append(base)
    return out


def violations() -> set[str]:
    stems = _dotted_aliases()
    found = set()
    for path in sorted(PKG.rglob("*.py")):
        rel = path.relative_to(PKG).as_posix()
        if rel.startswith("runtime/equivalent/"):
            continue
        src = path.read_text()
        tree = ast.parse(src, filename=str(path))
        modname = _module_name(path)
        me = _pkg_of(modname)
        if me is not None and me.endswith(".py"):
            me = me[:-3]
        is_pkg = path.name == "__init__.py"
        for m in re.finditer(r"parents\[\s*(\d+)\s*\]", src):
            if int(m.group(1)) >= 2:      # any repo-relative climb out of the package
                found.add(f"{rel}::parents[{m.group(1)}]")
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                f = node.func
                if (isinstance(f, ast.Attribute) and f.attr in ("insert", "append")
                        and isinstance(f.value, ast.Attribute) and f.value.attr == "path"
                        and isinstance(f.value.value, ast.Name) and f.value.value.id in ("sys", "_sys")):
                    found.add(f"{rel}::sys.path.{f.attr}")
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            for name in _imported(node, modname, is_pkg):
                top = name.split(".")[0]
                if top != "vqlab":
                    if top in stems and not (isinstance(node, ast.ImportFrom) and node.level):
                        found.add(f"{rel}::bare:{top}")
                        target = _pkg_of(stems[top])
                    else:
                        continue
                else:
                    canon = stems.get(".".join(name.split(".")[:2]), name)
                    target = _pkg_of(canon)
                if target is None or target in EXEMPT or me in EXEMPT:
                    continue
                if target == me or (me, target) in ALLOWED_PEERS:
                    continue
                lo, hi = LAYER.get(me), LAYER.get(target)
                if lo is None or hi is None:
                    found.add(f"{rel}::unlayered:{me}->{target}")
                elif hi >= lo:
                    found.add(f"{rel}::{'peer' if hi == lo else 'upward'}:{me}->{target}")
    return found


def _allowed() -> set[str]:
    return set(json.loads(ALLOWLIST.read_text())["violations"])


def test_no_new_layering_violations():
    new = sorted(violations() - _allowed())
    assert not new, ("new import-rule violations (see the docstring; fix the "
                     "import, do not extend the allowlist):\n  " + "\n  ".join(new))


def test_allowlist_only_shrinks():
    gone = sorted(_allowed() - violations())
    assert not gone, ("these allowlisted violations are fixed; regenerate the "
                      "allowlist (`python tests/test_layering.py --write`):\n  "
                      + "\n  ".join(gone))


def test_every_package_has_a_layer():
    pkgs = {p.name for p in PKG.iterdir() if p.is_dir() and (p / "__init__.py").exists()}
    pkgs |= {p.stem for p in PKG.glob("*.py") if p.stem not in ("__init__",)}
    missing = sorted(pkgs - set(LAYER) - EXEMPT)
    assert not missing, f"give these a layer in LAYER: {missing}"


def test_every_code_folder_is_a_package_with_its_context():
    """Nested sub-packages too (core/artifacts/ once had neither file):
    every folder with .py files has __init__.py, and every top-level package
    or routed sub-package (`<name>/CONTEXT.md` named by its parent) has a
    CONTEXT.md. Same rule the selftest runs."""
    from vqlab.selftest.layout_checks import packages_missing_context
    bad = packages_missing_context(PKG)
    assert not bad, f"add __init__.py / CONTEXT.md: {bad}"


# Tests whose PURPOSE is the old (bare / pre-split) names: they check that
# the alias finder hands back the canonical module object.
TESTS_ALIAS_ALLOWED = {"test_public_imports.py"}


def test_tests_use_canonical_imports():
    """tests/ imports modules where they live: no bare stems (`import
    vq_switch`) and no pre-split spellings (`from vqlab import vq_pack`)."""
    stems = _dotted_aliases()
    bad = []
    for path in sorted((ROOT / "tests").rglob("*.py")):
        if path.name in TESTS_ALIAS_ALLOWED:
            continue
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                names = [node.module or ""]
                if node.module == "vqlab":
                    names += [f"vqlab.{a.name}" for a in node.names]
            else:
                continue
            for name in names:
                parts = name.split(".")
                if parts[0] in stems:
                    bad.append(f"{path.name}:{node.lineno} {name}")
                elif (parts[0] == "vqlab" and len(parts) > 1
                      and stems.get(parts[1], f"vqlab.{parts[1]}") != f"vqlab.{parts[1]}"):
                    bad.append(f"{path.name}:{node.lineno} {name} -> {stems[parts[1]]}")
    assert not bad, "import these where they live:\n  " + "\n  ".join(bad)


if __name__ == "__main__" and "--write" in sys.argv:
    ALLOWLIST.parent.mkdir(exist_ok=True)
    ALLOWLIST.write_text(json.dumps({
        "_doc": "Known import-rule violations (tests/test_layering.py). Shrinks only.",
        "violations": sorted(violations())}, indent=1) + "\n")
    print(f"wrote {ALLOWLIST} ({len(violations())} entries)")
