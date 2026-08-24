#!/usr/bin/env python3
"""vqlab check — run the release gates that need no source model.

Composite of `check-release` (every file a downloader needs exists and
functions) and `check-bundle` (the shipped runtime matches this repo's).
Runs both, reports both, and fails if either fails — never stops at the
first failure, because knowing only that something broke is less useful
than knowing everything that broke.

NOT included, deliberately, because each needs an input this command does
not have:
  - the outlier gate       (`vqlab verify`, needs the bf16 source)
  - generation             (`vqlab smoke`, needs the model resident)
  - bundled-kernel accept  (`vqlab bundle-accept`)
  - comparator parity      (`vqlab check-comparator`, needs the teacher)
A pass here means the artifact is well-formed and ships the right runtime.
It does NOT mean the artifact is correct or that it can serve.

    vqlab check <artifact>
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).parent
GATES = [("check-release", "check_release.py"), ("check-bundle", "check_bundle.py")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vqlab check",
                                 description=__doc__.split("\n")[0])
    ap.add_argument("artifact")
    a = ap.parse_args(argv)

    results = []
    for name, script in GATES:
        print(f"\n=== {name} ===", flush=True)
        p = subprocess.run([sys.executable, str(HERE / script),
                            "--artifact", a.artifact])
        results.append((name, p.returncode == 0))

    print("\n=== summary ===")
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    failed = [n for n, ok in results if not ok]
    if failed:
        print(f"\n{len(failed)} gate(s) failed: {', '.join(failed)}")
        return 1
    print("\nAll release gates passed. Still required before release: "
          "`vqlab verify` (outlier gate, on a box that did not fit it) and "
          "`vqlab smoke` (one token through the shipping runtime).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
