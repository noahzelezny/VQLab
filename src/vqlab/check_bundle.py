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

ABSENT vs DRIFTED (2026-09-09). Verbatim containment answers "is this the
CURRENT text", not "is the runtime here at all", and the dense branch used
to report both as the ModuleNotFoundError case. It was wrong on all three
27B rungs and gemma-e4b-PLE: each carries vq_switch inline (VQSwitchLinear,
_dense_fused, _fused and _resolve_kernel are all DEFINED in the bundle) and
differs from the repo only by the 62 lines of RTILE/CB_DEV work that landed
after they were spliced. That is ordinary drift, the same failure the other
15 artifacts have — not a stock-install break. So absence is now decided by
whether the runtime's top-level defs are defined in the bundle, and only
that case claims ModuleNotFoundError.
"""
import argparse
import json
import pathlib
import re
import sys


# The symbols a dense bundle's fused path actually reaches for. If these are
# DEFINED in model.py, _resolve_kernel finds them in its own globals (tier 1)
# and never touches mlx_lm.models.vq_switch -- so the artifact loads on a
# stock install regardless of how far the rest of the text has drifted.
# Do NOT widen this to "every top-level def": a kernel added to the repo
# AFTER an artifact was spliced (gemmseg_cb_dev, 2026-09-08) is absent from
# every older bundle, and reporting that as ModuleNotFoundError is exactly
# the false alarm this function exists to end. Missing-and-uncalled is drift.
_DENSE_ANCHORS = {
    "vq_switch.py": ("VQSwitchLinear", "_dense_fused", "_fused"),
    "vq_dense.py": ("VQLinear", "VQEmbedding", "_decode_matmul",
                    "_resolve_kernel"),
}


def _undefined_anchors(name: str, bundle: str) -> list:
    """Load-bearing symbols of `name` that `bundle` never defines."""
    return [a for a in _DENSE_ANCHORS.get(name, ())
            if not re.search(rf"^(?:def|class)\s+{re.escape(a)}\b", bundle, re.M)]


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
        absent, drifted = [], []
        for name, text in (("vq_dense.py", vd), ("vq_switch.py", vs)):
            if text in bundle:
                continue
            undefined = _undefined_anchors(name, bundle)
            (absent if undefined else drifted).append((name, undefined))
        if absent:
            detail = "; ".join(
                f"{n} (undefined: {', '.join(u)})" for n, u in absent)
            print(f"FAIL (dense artifact): bundle is missing {detail}. "
                  "Its fused path would resolve against site-packages, so the "
                  "artifact requires a VQ-patched mlx-lm and raises "
                  "ModuleNotFoundError on a stock install. Re-run build-dense "
                  "to write a bundle carrying both runtimes.")
            return 1
        if drifted:
            names = ", ".join(n for n, _ in drifted)
            print(f"FAIL (dense artifact): bundled model.py "
                  f"({len(bundle.splitlines())} lines) carries every top-level "
                  f"def of {names}, but not the CURRENT text. This is drift, "
                  "not a missing runtime \u2014 the artifact loads on a stock "
                  "install and runs code older than the benches. Re-splice "
                  "before publishing any claim.")
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
