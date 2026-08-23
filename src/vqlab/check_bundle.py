#!/usr/bin/env python3
"""Gate: an artifact's bundled model.py must carry the CURRENT repo runtime.

External users run the bundle; benches run the venv runtime. Any drift means
published speed/quality claims describe code downloaders don't have.

MoE artifacts (config declares vq_modules): PASS = repo vq_switch.py text is
contained verbatim in the bundle.

Dense artifacts (config declares vq_linear/vq_embed): the runtime is
vq_dense.py, whose fused path needs _dense_fused/_fused from vq_switch.py,
so a correct dense bundle carries BOTH files' text (build_dense_vq.py
writes them in that order, followed by the loader shim). Dense bundles that
carry vq_dense.py ALONE reach into `mlx_lm.models.vq_switch` at call time
and therefore require a VQ-patched mlx-lm: measured, such an artifact
raises ModuleNotFoundError on a stock install — it scores fine and cannot
serve. Passing this gate is still not proof of which copy executes; see
METHODOLOGY.md §5 and run bundle-accept.
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
                  "Its fused path would resolve against site-packages, so the "
                  "artifact requires a VQ-patched mlx-lm and raises "
                  "ModuleNotFoundError on a stock install. Re-run build-dense "
                  "to write a bundle carrying both runtimes.")
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
