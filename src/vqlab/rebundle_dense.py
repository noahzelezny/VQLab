#!/usr/bin/env python3
"""Re-splice a DENSE artifact's model.py from the current runtime, in place.

`bundle` (add_model_file.py) is the MoE bundler and refuses dense
artifacts — it would drop vq_dense.py and leave a runtime that cannot
serve. `build-dense` rewrites a whole artifact from a base + fits, which
is the wrong tool when only the runtime moved. This is the missing
middle: same concatenation build_dense_vq.py ends with (vq_switch.py +
vq_dense.py + the loader shim), written over an existing artifact, with
a backup and a compile check.

    vqlab rebundle-dense --artifact <dir> [--backup-suffix .pre-resync]
"""
import argparse
import json
import pathlib

HERE = pathlib.Path(__file__).parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--backup-suffix", default=".pre-resync",
                    help="suffix for the previous model.py (empty = none)")
    a = ap.parse_args()
    art = pathlib.Path(a.artifact)
    cfg_path = art / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"no config.json in {art}")
    cfg = json.loads(cfg_path.read_text())
    if not (cfg.get("vq_linear") or cfg.get("vq_embed")):
        raise SystemExit(
            "REFUSING: this artifact carries no vq_linear/vq_embed, so it is "
            "NOT dense — use `vqlab bundle` (the MoE bundler) instead.")

    from vqlab.dense_shim import SHIM
    if "class Model" not in SHIM or "VQLinear" not in SHIM:
        raise SystemExit("FAIL: dense shim is missing class Model / VQLinear.")
    model_py = ((HERE / "vq_switch.py").read_text()
                + (HERE / "vq_dense.py").read_text() + SHIM)
    # never ship a model.py that cannot parse (build_dense_vq's own rule)
    compile(model_py, "model.py", "exec")

    dst = art / "model.py"
    if a.backup_suffix and dst.exists():
        (art / f"model.py{a.backup_suffix}").write_text(dst.read_text())
    dst.write_text(model_py)
    if cfg.get("model_file") != "model.py":
        cfg["model_file"] = "model.py"
        cfg_path.write_text(json.dumps(cfg, indent=1))
    print(f"dense bundle rewritten ({len(model_py.splitlines())} lines): {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
