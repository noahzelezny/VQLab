#!/usr/bin/env python3
"""Gate: an artifact's bundled model.py must carry the CURRENT repo runtime.

External users run the bundle; our benches run the venv runtime. Any drift
means published speed/quality claims describe code downloaders don't have.
PASS = repo vq_switch.py text is contained verbatim in the bundle.
"""
import argparse, pathlib, sys
ap = argparse.ArgumentParser()
ap.add_argument("--artifact", required=True)
a = ap.parse_args()
runtime = (pathlib.Path(__file__).parent / "vq_switch.py").read_text()
bp = pathlib.Path(a.artifact) / "model.py"
if not bp.exists():
    print("FAIL: no bundled model.py"); sys.exit(1)
bundle = bp.read_text()
if runtime in bundle:
    print(f"PASS: bundle carries the current runtime ({len(runtime.splitlines())} lines) verbatim")
    sys.exit(0)
print(f"FAIL: bundled model.py ({len(bundle.splitlines())} lines) does not contain the "
      f"current runtime ({len(runtime.splitlines())} lines). Downloaders run different "
      f"code than the benches. Re-splice before publishing any claim.")
sys.exit(1)
