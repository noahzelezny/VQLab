"""vqlab — one CLI over the standalone pipeline scripts.

The underlying scripts are deliberately standalone (each is a complete,
auditable tool with its own argparse surface, and several run at module
scope). This dispatcher sets sys.argv and executes the chosen script in
its own right, so `vqlab fit-moe --help` shows the script's full surface
and behavior is identical to running the script directly.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

PKG = Path(__file__).parent

COMMANDS = {
    # fitting
    "fit-moe": ("vq_397b_codes.py", "fit VQ codebooks for MoE expert tensors (per --family)"),
    "fit-dense": ("fit_dense_vq.py", "fit VQ codebooks for a dense MLP trio"),
    # packing
    "pack": ("pack_artifact.py", "pack MoE codes to true bit-width; recompute sizes"),
    "pack-ple": ("pack_ple.py", "pack PLE codes row-aligned to true bit-width"),
    "stream-convert": ("stream_convert.py", "streaming affine convert / struct base for models bigger than RAM"),
    "splice-ple": ("splice_ple.py", "splice VQ PLE codes into a packed artifact"),
    "pack-dense": ("pack_dense.py", "pack a dense VQ artifact"),
    # assembly
    "build-dense": ("build_dense_vq.py", "splice dense VQ fits into a quantized base -> runnable artifact"),
    "graft": ("graft_vision.py", "graft the bf16 vision tower into an artifact"),
    "bundle": ("add_model_file.py", "(re)write the self-contained model.py bundle (MoE)"),
    # gates
    "check": ("check_all.py", "run the release gates that need no source model"),
    "smoke": ("smoke.py", "generate one token through the runtime the artifact ships"),
    "verify": ("verify_artifact.py", "outlier gate: decode artifact bytes vs bf16 source"),
    "check-release": ("check_release.py", "release gate: files exist and function"),
    "check-bundle": ("check_bundle.py", "bundle gate: shipped runtime matches repo runtime"),
    "check-comparator": ("check_comparator.py", "comparator gate: tensor-set parity vs teacher"),
    "bundle-accept": ("bundle_accept.py", "kernel acceptance on the runtime lifted FROM the artifact"),
    "manifest": ("artifact_manifest.py", "write/check provenance manifests"),
    "preflight-ram": ("preflight_ram.py", "refuse resident-memory ops on models bigger than RAM"),
    "preflight-disk": ("preflight_disk.py", "refuse builds whose output volume lacks the space"),
    # scoring
    "score": ("referee/score_streaming.py", "streaming referee perplexity (models may exceed RAM)"),
    "kl": ("kl_damage.py", "KL-to-bf16 damage vs a cached teacher (cache/score)"),
    # verification
    "selftest": ("selftest.py", "run the real pipeline on a tiny synthetic model"),
    # planning / probes
    "price": ("price.py", "price a size-targeted build before fitting it"),
    "probe-init": ("probe_init_sweep.py", "per-family k-means++ vs random init sweep"),
}


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print("vqlab — size-targeted VQ quantization for MLX\n\ncommands:")
        for name, (_, desc) in COMMANDS.items():
            print(f"  {name:18s} {desc}")
        print("\n`vqlab <command> --help` shows each command's full surface.")
        print("Read METHODOLOGY.md before publishing any number.")
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd not in COMMANDS:
        print(f"unknown command: {cmd}", file=sys.stderr)
        return 2
    script = PKG / COMMANDS[cmd][0]
    sys.argv = [str(script), *rest]
    sys.path.insert(0, str(PKG))  # sibling imports (vq_pack, vq_switch)
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
