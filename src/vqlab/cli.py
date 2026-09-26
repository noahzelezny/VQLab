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
    # plan/
    "onboard": ("onboard.py", "sequence a new teacher through ONBOARDING.md: profile, loader, cache determinism, init sweep (resumable; --launch runs GPU steps under the lease)"),
    "family-profile": ("family_profile.py", "size up a teacher from headers only: arch, bytes per bit, legal (d,K), module signatures"),
    "preflight-ram": ("preflight_ram.py", "refuse resident-memory ops on models bigger than RAM"),
    "preflight-disk": ("preflight_disk.py", "refuse builds whose output volume lacks the space"),
    "price": ("price.py", "price a size-targeted build before fitting it"),
    "layer-leverage": ("layer_leverage.py", "per-layer damage probe: which layers earn bigger K"),
    "alloc-sweep": ("alloc_sweep.py", "cost/value curves for per-layer allocation -> the iso-byte frontier"),
    "probe-init": ("probe_init_sweep.py", "per-family k-means++ vs random init sweep"),
    "mtp-probe": ("mtp_probe.py", "MTP head draft-acceptance probe (qwen4_exp)"),
    "mtp-probe35": ("mtp_probe35.py", "MTP wiring sweep + acceptance probe (qwen3_5 / qwen3_5_moe)"),
    # fit/
    "fits": ("fits.py", "the fit store: index / list / file fitted modules by family, teacher, layer, geometry"),
    "fit-moe": ("vq_397b_codes.py", "fit VQ codebooks for MoE expert tensors (per --family)"),
    "fit-dense": ("fit_dense_vq.py", "fit VQ codebooks for a dense MLP trio"),
    "fit-ple": ("fit_ple.py", "fit per-tensor VQ codebooks for PLE ngram banks"),
    "geo-build": ("geo_build.py", "rebuild an artifact under a new per-layer geometry map (diff-style)"),
    "harvest-parts": ("harvest_parts.py", "extract a shipped rung's VQ fits into a geo-build --reuse parts dir"),
    # assemble/
    "pack": ("pack_artifact.py", "pack MoE codes to true bit-width; recompute sizes"),
    "pack-ple": ("pack_ple.py", "pack PLE codes row-aligned to true bit-width"),
    "stream-convert": ("stream_convert.py", "streaming affine convert / struct base for models bigger than RAM"),
    "splice-ple": ("splice_ple.py", "splice VQ PLE codes into a packed artifact"),
    "pack-dense": ("pack_dense.py", "pack a dense VQ artifact"),
    "unpack-dense": ("unpack_dense.py", "diagnostic twin with PLAIN codes: isolates PACKING from every other variable (F141)"),
    "build-dense": ("build_dense_vq.py", "splice dense VQ fits into a quantized base -> runnable artifact"),
    "graft": ("graft_vision.py", "graft the bf16 vision tower into an artifact"),
    "ple-swap": ("ple_swap.py", "swap an artifact's PLE tables for another rung's (symlinks; prices PLE bytes on KL)"),
    "mtp-extract": ("mtp_extract.py", "pull a model's MTP head out of its source checkpoint into a graft"),
    "mtp-pack": ("mtp_pack.py", "pack a bf16 MTP graft into a quantized drafting sidecar"),
    "mtp-graft": ("mtp_graft.py", "emit the MTP head in a native runtime's layout (language_model.mtp.*)"),
    # bundle/
    "bundle": ("add_model_file.py", "(re)write the self-contained model.py bundle (MoE)"),
    "patch-arch": ("patch_arch.py", "surgical loader-aware arch fix on a published bundle, runtime untouched"),
    "vision-layout": ("vision_layout.py", "check/repair a grafted tower's patch-embed memory layout"),
    "rebundle-dense": ("rebundle_dense.py", "re-splice a DENSE artifact's model.py from the current runtime, in place"),
    # gate/
    "vision-smoke": ("vision_smoke.py", "vision arm of III.11: put an IMAGE through the shipping runtime"),
    "check": ("check_all.py", "run the release gates that need no source model"),
    "smoke": ("smoke.py", "generate one token through the runtime the artifact ships"),
    "verify": ("verify_artifact.py", "outlier gate: decode artifact bytes vs bf16 source"),
    "validate": ("validate_queue.py", "overnight validation queue: drain artifacts through the gates, never publish"),
    "check-release": ("check_release.py", "release gate: files exist and function"),
    "check-bundle": ("check_bundle.py", "bundle gate: shipped runtime matches repo runtime"),
    "check-comparator": ("check_comparator.py", "comparator gate: tensor-set parity vs teacher"),
    "bundle-accept": ("bundle_accept.py", "kernel acceptance on the runtime lifted FROM the artifact"),
    "selftest": ("selftest.py", "run the real pipeline on a tiny synthetic model"),
    "mtp-smoke-head": ("mtp_smoke_head.py", "head-alone load + T=1/T=2 forward timing (no trunk)"),
    # score/
    "tasks": ("score_tasks_streaming.py", "task benchmarks (HellaSwag/PIQA/WinoGrande via lm-eval), layer-streamed: scores models larger than RAM"),
    "score": ("referee/score_streaming.py", "streaming referee perplexity (models may exceed RAM)"),
    "kl": ("kl_damage.py", "KL-to-bf16 damage vs a cached teacher (cache/score)"),
    "kernel-truth": ("kernel_truth.py", "fused vs fallback: which is closer to an EXACT float64 reference? (F146)"),
    "kl-pair": ("kl_pair.py", "paired KL between two arms scored in SEPARATE runs (env-var arms)"),
    "kl-ladder": ("kl_ladder.py", "rank rungs by KL against per-corpus teacher caches, with error bars"),
    # bench/
    "host-attrib": ("host_attrib.py", "profile the VQ module's HOST cost (F41's unattributed 12-13%)"),
    "prefill-bench": ("prefill_bench.py", "time prefill on one artifact under BOTH codebook arms (ratio, n>=3)"),
    "active-bytes": ("active_bytes.py", "weight bytes read per decode token, by component (the roofline denominator)"),
    "coverage": ("kernel_coverage.py", "per-module report: which prefill kernel path each geometry takes"),
    "hc-micro": ("hc_micro.py", "single GatedResidual micro-bench: is mx.compile inert on this chain? (split half invalid, see F133)"),
    "decode-timeline": ("decode_timeline.py", "every stage of one decode token, in order, timed, summing to the whole"),
    "decode-ladder": ("decode_ladder.py", "per-component decode deletion arms (GDN / attn / VQ), Flash-aware"),
    "mtp-accept": ("mtp_accept.py", "paired draft-acceptance across prompts (the reliable instrument)"),
    "mtp-bench": ("mtp_bench.py", "measure the MTP speedup, acceptance and numerics control"),
    # records/
    "manifest": ("artifact_manifest.py", "tamper stamp: were these shard bytes rewritten? (outside the artifact)"),
    "runs": ("runlog.py", "the run log: every vqlab invocation, argv, commit, exit code"),
    "provenance": ("provenance.py", "build record: how an artifact was made (tool, fitter, inputs, runtime, lineage)"),
    # ship/
    "publish": ("publish.py", "upload to the Hub, gated on check-release passing"),
    "mtp-generate": ("mtp_run.py", "generate with MTP speculative drafting"),
    "serve": ("serve.py", "serve an artifact over an OpenAI-compatible API (mlx-lm server + MTP decode)"),
    # agents/
    "mcp": ("mcp_server.py", "serve the lab to agents over MCP (stdio); one server per box"),
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
    from vqlab import _layout   # stage dirs on sys.path; old dotted names aliased
    script = _layout.find(COMMANDS[cmd][0])
    sys.argv = [str(script), *rest]
    import runlog
    run = runlog.start(cmd, rest)
    rc = 0
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        raise
    except BaseException as e:
        rc = f"{type(e).__name__}: {str(e)[:200]}"
        raise
    finally:
        runlog.end(run, rc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
