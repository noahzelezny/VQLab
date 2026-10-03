#!/usr/bin/env python
"""parity — a family's checklist of REFERENCE behaviours, tested or not.

Each family plugin (vqlab/family/<name>.py) may declare `reference` (where
the official inference code lives) and `parity`: behaviours of that code the
runtime must reproduce (clamps, norms and eps, routing bias, hash layers),
each naming the test that checks it. F195 is why: DeepSeek's reference
clamps the shared expert's SwiGLU, mlx-lm did not, and nobody noticed until
a session read inference/model.py.

This prints the checklist and exits 1 if a REQUIRED item has no test, or
names a test that does not exist (file or function), or whose test does not
PASS when run here. The named tests are run (pytest, tiny CPU fixtures): a
test that SKIPS -- e.g. the architecture is missing from this interpreter --
proves nothing, so it counts as untested (it did, until 2026-10-03).
--no-run only checks that the tests exist. No model.

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


def run_tests(specs, root: pathlib.Path) -> dict:
    """{spec: 'passed' | 'skipped' | 'failed' | 'error'} from one pytest run."""
    import subprocess
    if not specs:
        return {}
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rA", "-p", "no:cacheprovider",
                        "-m", "lab or not lab", *sorted(set(specs))],
                       cwd=root, capture_output=True, text=True)
    got = {}
    for line in r.stdout.splitlines():
        m = re.match(r"^(PASSED|SKIPPED|FAILED|ERROR)\s+(\S+?)(?:\s|$)", line)
        if m:
            got[m.group(2).split("[")[0]] = m.group(1).lower()
    # SKIPPED lines name the file:line, not the function: a skipped module skips all of it
    for line in r.stdout.splitlines():
        m = re.match(r"^SKIPPED \[\d+\] (\S+?):\d+", line)
        if m:
            for sp in specs:
                if sp.partition("::")[0] == m.group(1) and sp not in got:
                    got[sp] = "skipped"
    return {sp: got.get(sp, "error") for sp in specs}


def check(plugin, root: pathlib.Path, out=print, results=None) -> int:
    """Print one family's checklist; return the number of failures."""
    out(f"{plugin.name}: reference = {plugin.reference or '(not declared)'}")
    if not plugin.parity:
        out("  (no parity items declared)")
        return 0
    bad = 0
    for it in plugin.parity:
        res = (results or {}).get(it.test)
        if it.test and has_test(it.test, root) and results is not None and res != "passed":
            st = f"TEST-{(res or 'error').upper()}"
            bad += 1 if it.required or res in ("failed", "error") else 0
        elif it.test and has_test(it.test, root):
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
    ap.add_argument("--no-run", action="store_true",
                    help="only check the named tests exist; do not run them")
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
    results = None if a.no_run else run_tests(
        [it.test for p in ps for it in p.parity if it.test and has_test(it.test, root)], root)
    bad = sum(check(p, root, results=results) for p in ps)
    print(f"{'FAIL' if bad else 'PASS'}: {bad} required item(s) without a passing test"
          + (" (tests not run: --no-run)" if a.no_run else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
