#!/usr/bin/env python
"""parity — a family's checklist of REFERENCE behaviours, tested or not.

Each family plugin (vqlab/family/<name>.py) may declare `reference` (where
the official inference code lives) and `parity`: behaviours of that code the
runtime must reproduce (clamps, norms and eps, routing bias, hash layers),
each naming the test that checks it. F195 is why: DeepSeek's reference
clamps the shared expert's SwiGLU, mlx-lm did not, and nobody noticed until
a session read inference/model.py.

This prints the checklist and exits 1 if a REQUIRED item has no test, or
names a test that does not exist (file or function). It does not run the
tests; `pytest <test>` does. No GPU, no model.

    vqlab parity deepseek_v4
    vqlab parity --all
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys


def has_test(spec: str, root: pathlib.Path) -> bool:
    """'tests/f.py::fn' names a file under root that defines fn."""
    path, _, fn = spec.partition("::")
    f = root / path
    if not f.is_file() or not fn:
        return False
    return re.search(rf"^def {re.escape(fn)}\(", f.read_text(), re.M) is not None


def check(plugin, root: pathlib.Path, out=print) -> int:
    """Print one family's checklist; return the number of failures."""
    out(f"{plugin.name}: reference = {plugin.reference or '(not declared)'}")
    if not plugin.parity:
        out("  (no parity items declared)")
        return 0
    bad = 0
    for it in plugin.parity:
        if it.test and has_test(it.test, root):
            st = "tested"
        elif it.test:
            st, bad = "MISSING-TEST", bad + 1
        elif it.required:
            st, bad = "UNTESTED-REQUIRED", bad + 1
        else:
            st = "untested"
        out(f"  [{st:17s}] {it.name}{'  (act-stats: ' + it.probe + ')' if it.probe else ''}")
        out(f"      reference: {it.reference}")
        out(f"      runtime:   {it.runtime}")
        if it.test:
            out(f"      test:      {it.test}")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vqlab parity", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("family", nargs="?", help="family plugin name or model_type")
    ap.add_argument("--all", action="store_true", help="every family plugin")
    ap.add_argument("--root", default=None,
                    help="repo root the test paths resolve against (default: this checkout)")
    a = ap.parse_args(argv)
    from vqlab.family import get, plugins
    root = pathlib.Path(a.root) if a.root else pathlib.Path(__file__).resolve().parents[3]
    if a.all:
        ps = list(plugins().values())
    elif a.family:
        p = get(a.family)
        if p is None:
            sys.exit(f"FAIL: no family plugin {a.family!r} (have {sorted(plugins())})")
        ps = [p]
    else:
        ap.error("name a family or pass --all")
    if not (root / "tests").is_dir():
        sys.exit(f"FAIL: no tests/ under {root}; pass --root <checkout> "
                 "(an installed package carries no tests)")
    bad = sum(check(p, root) for p in ps)
    print(f"{'FAIL' if bad else 'PASS'}: {bad} required item(s) without a test")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
