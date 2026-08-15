#!/usr/bin/env python
"""Install the VQ runtime into an mlx_lm site-packages tree (M1d).

Idempotent: copies quantlab/vq_switch.py -> mlx_lm/models/vq_switch.py and
inserts the loader hook into utils.load_model just before
`model.load_weights`. Re-run after editing vq_switch.py or after an mlx_lm
upgrade. --check reports state without writing.

Usage:
    python patch_mlx_lm.py [--check] [--site PATH_TO/mlx_lm]
(default --site: the mlx_lm importable from THIS interpreter)
"""
import argparse
import pathlib
import shutil
import sys

MARK = "# --- quantlab VQ hook (patch_mlx_lm.py) ---"
HOOK = f"""
    {MARK}
    _vq_prefixes = sorted({{k[:-6] for k in weights if k.endswith(".codes")}})
    if _vq_prefixes:
        from mlx.utils import tree_unflatten
        from .models.vq_switch import VQSwitchLinear
        model.update_modules(tree_unflatten([
            (p, VQSwitchLinear.from_weights(
                weights[p + ".codes"], weights[p + ".codebook"],
                weights[p + ".vq_scales"]))
            for p in _vq_prefixes
        ]))
    # --- end quantlab VQ hook ---
"""
ANCHOR = "    model.eval()\n    model.load_weights(list(weights.items()), strict=strict)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", help="path to mlx_lm package dir")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if args.site:
        pkg = pathlib.Path(args.site)
    else:
        import mlx_lm
        pkg = pathlib.Path(mlx_lm.__file__).parent
    utils = pkg / "utils.py"
    dst = pkg / "models" / "vq_switch.py"
    src = pathlib.Path(__file__).parent / "vq_switch.py"
    text = utils.read_text()

    hooked = MARK in text
    module_current = dst.exists() and dst.read_text() == src.read_text()
    print(f"mlx_lm at {pkg}")
    print(f"  vq_switch.py installed+current: {module_current}")
    print(f"  load_model hook present:        {hooked}")
    if args.check:
        return 0 if (hooked and module_current) else 1

    if not module_current:
        shutil.copy2(src, dst)
        print(f"  -> copied {src.name}")
    if not hooked:
        if ANCHOR not in text:
            print("  !! anchor not found in utils.py — mlx_lm layout changed, "
                  "patch by hand")
            return 2
        utils.write_text(text.replace(ANCHOR, HOOK + ANCHOR, 1))
        print("  -> hook inserted before model.load_weights")
    return 0


if __name__ == "__main__":
    sys.exit(main())
