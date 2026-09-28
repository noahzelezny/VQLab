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
    "zero-groups": ("zero_groups.py", "headers + vq_scales only, no GPU: code GiB a skip-zero-groups format would save, per layer / per fit"),
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
    "reselect": ("reselect.py", "activation-aware code re-selection: calibrate (self-gen Gram bank) / apply (new artifact, codes only)"),
    "fit-additive": ("additive_vq.py", "additive VQ: two small codebooks per module, expanded to one d4-K(K1*K2) fit for zero-kernel KL tests"),
    "harvest-parts": ("harvest_parts.py", "extract a shipped rung's VQ fits into a geo-build --reuse parts dir"),
    # assemble/
    "pack": ("pack_artifact.py", "pack MoE codes to true bit-width; recompute sizes"),
    "pack-ple": ("pack_ple.py", "pack PLE codes row-aligned to true bit-width"),
    "stream-convert": ("stream_convert.py", "streaming affine convert / struct base for models bigger than RAM"),
    "splice-ple": ("splice_ple.py", "splice VQ PLE codes into a packed artifact"),
    "pack-dense": ("pack_dense.py", "pack a dense VQ artifact"),
    "unpack-dense": ("unpack_dense.py", "diagnostic twin with PLAIN codes: isolates PACKING from every other variable (F141)"),
    "build-dense": ("build_dense_vq.py", "splice dense VQ fits into a quantized base -> runnable artifact"),
    "reskeleton": ("reskeleton.py", "put a VQ build's expert tensors on another build's skeleton (matched-skeleton comparisons)"),
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
    "spelling": ("us_spelling.py", "fail on British spellings in released text (paper, cards); --fix rewrites to US"),
    "pin": ("pin.py", "freeze an artifact for measurement (symlinked weights, optional --runtime re-bake), smoke it, write vqlab_pin.json"),
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
    "stream-score": ("stream_score.py", "layer-streamed ppl / teacher top-k cache / KL-to-teacher: the KL gate's instrument (kl-ladder runs it per rung)"),
    "kl": ("kl_damage.py", "KL-to-bf16 damage vs a cached teacher (cache/score)"),
    "kernel-truth": ("kernel_truth.py", "fused vs fallback: which is closer to an EXACT float64 reference? (F146)"),
    "kl-pair": ("kl_pair.py", "paired KL between two arms scored in SEPARATE runs (env-var arms)"),
    "kl-ladder": ("kl_ladder.py", "rank rungs by KL against per-corpus teacher caches, with error bars"),
    # bench/
    "speed-pair": ("speed_pair.py", "decode/prefill speed of TWO artifacts as a per-pair ratio (fresh process per arm, alternating, n>=3)"),
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
    "registry": ("registry.py", "index of every artifact from provable facts; Hub drift check (metadata only)"),
    # ship/
    "publish": ("publish.py", "upload to the Hub, gated on check-release passing"),
    "mtp-generate": ("mtp_run.py", "generate with MTP speculative drafting"),
    "serve": ("serve.py", "serve an artifact over an OpenAI-compatible API (mlx-lm server + MTP decode)"),
    # agents/
    "mcp": ("mcp_server.py", "serve the lab to agents over MCP (stdio); one server per box"),
    "gui": ("gui.py", "a local, read-only window onto the lab (127.0.0.1:8781)"),
    "queue": ("run_queue.py", "run a list of steps from PINNED code (git worktree), under the GPU lease, failing loudly; --preflight"),
}


# Build tools whose output is an ARTIFACT: after a clean exit the CLI writes
# its build record (records/provenance.py) unless the tool already wrote a
# richer one in this run (the fitters do). Candidates are tried in order; a
# FILE output is recorded in the artifact dir that holds it, and only a dir
# with a config.json counts, so a sidecar written to a scratch dir never
# stamps that dir. In-place tools produce an AMENDMENT: the prior record is
# kept in vqlab_provenance.history.jsonl and linked by id.
BUILD_OUTPUTS = {
    "pack": ("--out",), "pack-dense": ("--out",), "build-dense": ("--out",),
    "stream-convert": ("--out",), "ple-swap": ("--out",), "unpack-dense": ("--out",),
    "pack-ple": ("--artifact",), "graft": ("--artifact",), "splice-ple": ("--artifact",),
    "bundle": ("--artifact",), "rebundle-dense": ("--artifact",),
    "patch-arch": ("--out",), "vision-layout": ("--out", 0),
    "mtp-pack": ("--out", "--model"), "mtp-extract": ("--out",),
    "mtp-graft": ("--out", "--model"),
}


def _parse(argv):
    named, pos, i = {}, [], 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--"):
            if "=" in a:
                k, v = a.split("=", 1)
                named[k] = v
            elif i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                named[a] = argv[i + 1]
                i += 1
            else:
                named[a] = True
        else:
            pos.append(a)
        i += 1
    return named, pos


def _record_build(cmd, rest, script):
    """Write the build record for a build command's artifact. Never raises:
    a record failure is reported, and must not turn a finished build red."""
    import os
    try:
        named, pos = _parse(rest)
        if cmd == "vision-layout" and "--fix" not in named:
            return                                    # report-only mode
        target = None
        for c in BUILD_OUTPUTS[cmd]:
            v = pos[c] if isinstance(c, int) and len(pos) > c else named.get(c)
            if not isinstance(v, str):
                continue
            d = Path(v)
            d = d.parent if d.is_file() else d
            if d.is_dir() and (d / "config.json").exists():
                target = d
                break
        if target is None:
            return
        import provenance
        old = target / provenance.RECORD
        if old.exists():
            import json
            if json.loads(old.read_text())["tool"].get("run_id") == os.environ.get("VQLAB_RUN_ID"):
                return                                # the tool wrote its own
        inputs = []
        for k, v in list(named.items()) + [(f"arg{i}", v) for i, v in enumerate(pos)]:
            if isinstance(v, str) and v.startswith(("/", ".", "~")) and Path(v).exists() \
                    and Path(v).resolve() != target.resolve() \
                    and Path(v).resolve().parent != target.resolve():
                inputs.append((k.lstrip("-"), Path(v) if Path(v).is_dir() else Path(v).parent))
        provenance.write_build_record(
            target, tool=cmd, script=str(script), argv=[cmd, *rest], inputs=inputs,
            method={"recorded_by": "cli", "note": "this tool's settings are its argv"})
    except Exception as e:                           # noqa: BLE001
        print(f"vqlab: build record NOT written for {cmd}: {type(e).__name__}: {e}",
              file=sys.stderr)


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
        if rc == 0 and cmd in BUILD_OUTPUTS:
            _record_build(cmd, rest, script)
    return 0


if __name__ == "__main__":
    sys.exit(main())
