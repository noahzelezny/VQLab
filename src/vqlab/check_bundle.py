#!/usr/bin/env python3
"""Gate: an artifact's bundled model.py must carry the CURRENT repo runtime.

External users run the bundle; benches run the venv runtime. Any drift means
published speed/quality claims describe code downloaders don't have.

MoE artifacts (config declares vq_modules): PASS = repo vq_switch.py text is
contained verbatim in the bundle.

Dense artifacts (config declares vq_linear/vq_embed): the runtime is
vq_dense.py, which ALSO does a lazy `from mlx_lm.models.vq_switch import
_fused, _dense_fused` at call time — so a correct dense bundle must carry
BOTH files' text, and even then the import may resolve to site-packages
(see METHODOLOGY.md §5: instrument the import, never assume). There is no
dense bundle writer yet; this gate fails dense artifacts loudly rather than
letting the gap pass silently.
"""
import argparse
import json
import pathlib
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    a = ap.parse_args()
    here = pathlib.Path(__file__).parent
    art = pathlib.Path(a.artifact)

    cfg = {}
    cfg_path = art / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
    dense = bool(cfg.get("vq_linear") or cfg.get("vq_embed"))

    bp = art / "model.py"
    if not bp.exists():
        print("FAIL: no bundled model.py")
        return 1
    bundle = bp.read_text()

    if dense:
        vd = (here / "vq_dense.py").read_text()
        vs = (here / "vq_switch.py").read_text()
        missing = [n for n, t in (("vq_dense.py", vd), ("vq_switch.py", vs))
                   if t not in bundle]
        if missing:
            print(f"FAIL (dense artifact): bundle is missing {', '.join(missing)}. "
                  "No dense bundle writer exists yet — dense artifacts resolve "
                  "their fused path against site-packages, not the bundle. Do "
                  "not publish a dense artifact with a runtime-dependent claim "
                  "until the bundle carries both runtimes AND bundle_accept "
                  "passes on the lifted copy.")
            return 1
        print("PASS: dense bundle carries both runtimes verbatim (still "
              "instrument the resolved import before any runtime claim)")
        return 0

    runtime = (here / "vq_switch.py").read_text()
    if runtime in bundle:
        print(f"PASS: bundle carries the current runtime "
              f"({len(runtime.splitlines())} lines) verbatim")
        return 0
    print(f"FAIL: bundled model.py ({len(bundle.splitlines())} lines) does not "
          f"contain the current runtime ({len(runtime.splitlines())} lines). "
          "Downloaders run different code than the benches. Re-splice before "
          "publishing any claim.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
