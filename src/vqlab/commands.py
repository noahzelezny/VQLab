"""The command registry: every `vqlab <cmd>`, as plain data.

ONE table that the CLI (dispatch + help), the MCP server (its `run`
allowlist and GPU-free set) and the queue runner (name resolution) all read.
It is strings only and imports nothing from vqlab, so any layer may import
it without an edge (tests/test_layering.py puts it at layer 0).

    Command(name, module, file, stage, mcp, gpu, help)

  name    the flat command name: the canonical key (run log, queue steps,
          BUILD_OUTPUTS, box_quiet all speak it)
  module  the dotted module `vqlab <name>` runs (runpy.run_module)
  file    its pre-split filename, kept for cli.COMMANDS' old (file, help)
          view and _layout.find() callers
  stage   its CLI namespace (`vqlab <stage> <sub>`)
  mcp     an agent may launch it through MCP `run` (publish never is)
  gpu     it may touch the GPU; False = header reads / records / HTTP
          clients that never take the GPU lease (MCP GPU_FREE). Audited
          only for mcp=True commands; elsewhere True is the safe default
  help    one line for `vqlab --help`
"""
from __future__ import annotations

from typing import NamedTuple


class Command(NamedTuple):
    name: str
    module: str
    file: str
    stage: str
    mcp: bool
    gpu: bool
    help: str


_C = Command
COMMANDS: dict[str, Command] = {c.name: c for c in (
    _C('onboard', 'vqlab.agents.onboard', 'onboard.py', 'plan', True, False,
       'sequence a new teacher through ONBOARDING.md: profile, loader, cache determinism, init sweep (resumable; --launch runs GPU steps under the lease)'),
    _C('family-profile', 'vqlab.plan.family_profile', 'family_profile.py', 'plan', True, False,
       'size up a teacher from headers only: arch, bytes per bit, legal (d,K), module signatures'),
    _C('preflight-ram', 'vqlab.plan.preflight_ram', 'preflight_ram.py', 'plan', False, True,
       'refuse resident-memory ops on models bigger than RAM'),
    _C('preflight-disk', 'vqlab.plan.preflight_disk', 'preflight_disk.py', 'plan', False, True,
       'refuse builds whose output volume lacks the space'),
    _C('price', 'vqlab.plan.price', 'price.py', 'plan', True, False,
       'price a size-targeted build before fitting it'),
    _C('zero-groups', 'vqlab.plan.zero_groups', 'zero_groups.py', 'plan', False, True,
       'headers + vq_scales only, no GPU: code GiB a skip-zero-groups format would save, per layer / per fit'),
    _C('layer-leverage', 'vqlab.plan.layer_leverage', 'layer_leverage.py', 'plan', True, True,
       'per-layer damage probe: which layers earn bigger K'),
    _C('alloc-sweep', 'vqlab.plan.alloc_sweep', 'alloc_sweep.py', 'plan', True, True,
       'cost/value curves for per-layer allocation -> the iso-byte frontier'),
    _C('probe-init', 'vqlab.plan.probe_init_sweep', 'probe_init_sweep.py', 'plan', True, True,
       'per-family k-means++ vs random init sweep'),
    _C('mtp-probe', 'vqlab.plan.mtp_probe', 'mtp_probe.py', 'plan', False, True,
       'MTP head draft-acceptance probe (qwen4_exp)'),
    _C('mtp-probe35', 'vqlab.plan.mtp_probe35', 'mtp_probe35.py', 'plan', False, True,
       'MTP wiring sweep + acceptance probe (qwen3_5 / qwen3_5_moe)'),
    _C('doctor', 'vqlab.agents.doctor', 'doctor.py', 'lab', False, True,
       'which interpreter, mlx / mlx-lm / arch files, storage roots and HF token this run would use (no GPU)'),
    _C('config', 'vqlab.agents.config_cmd', 'config_cmd.py', 'lab', False, True,
       'show where vqlab reads/writes (and why), or `config init` a starter config'),
    _C('fits', 'vqlab.fit.fits', 'fits.py', 'fit', True, False,
       'the fit store: index / list / file fitted modules by family, teacher, layer, geometry'),
    _C('fit-moe', 'vqlab.fit.fit_moe', 'fit_moe.py', 'fit', True, True,
       'fit VQ codebooks for MoE expert tensors (per --family)'),
    _C('fit-dense', 'vqlab.fit.fit_dense_vq', 'fit_dense_vq.py', 'fit', True, True,
       'fit VQ codebooks for a dense MLP trio'),
    _C('fit-ple', 'vqlab.fit.fit_ple', 'fit_ple.py', 'fit', False, True,
       'fit per-tensor VQ codebooks for PLE ngram banks'),
    _C('geo-build', 'vqlab.fit.geo_build', 'geo_build.py', 'fit', True, True,
       'rebuild an artifact under a new per-layer geometry map (diff-style)'),
    _C('reselect', 'vqlab.fit.reselect', 'reselect.py', 'fit', False, True,
       'activation-aware code re-selection: calibrate (self-gen Gram bank) / apply (new artifact, codes only)'),
    _C('fit-additive', 'vqlab.fit.additive_vq', 'additive_vq.py', 'fit', False, True,
       'additive VQ: two small codebooks per module, expanded to one d4-K(K1*K2) fit for zero-kernel KL tests'),
    _C('harvest-parts', 'vqlab.fit.harvest_parts', 'harvest_parts.py', 'fit', False, True,
       "extract a shipped rung's VQ fits into a geo-build --reuse parts dir"),
    _C('pack', 'vqlab.build.pack', 'pack.py', 'build', False, True,
       'pack MoE codes to true bit-width; recompute sizes'),
    _C('pack-ple', 'vqlab.build.pack_ple', 'pack_ple.py', 'build', False, True,
       'pack PLE codes row-aligned to true bit-width'),
    _C('stream-convert', 'vqlab.build.stream_convert', 'stream_convert.py', 'build', False, True,
       'streaming affine convert / struct base for models bigger than RAM'),
    _C('sanitize-stream', 'vqlab.build.sanitize_stream', 'sanitize_stream.py', 'build', False, True,
       'exact runtime-layout teacher, one layer at a time through Model.sanitize (DeepSeek-V4)'),
    _C('loo-bands', 'vqlab.build.loo_bands', 'loo_bands.py', 'plan', False, True,
       'leave-one-out band hybrids of a VQ build (exact experts restored per band) + their kl-ladder queue: the damage map'),
    _C('minibase', 'vqlab.build.minibase', 'minibase.py', 'build', False, True,
       "a fit base of one layer band's shards (symlinks), so a band fit writes only those"),
    _C('mix', 'vqlab.build.mix', 'mix.py', 'build', False, True,
       'assemble an artifact from per-layer-band sources, shard by shard; config maps follow the bytes'),
    _C('release-prep', 'vqlab.ship.release_prep', 'release_prep.py', 'ship', False, True,
       'everything before an upload: sizes, junk, provenance, check-release, Hub diff; prints the publish line'),
    _C('card', 'vqlab.ship.card', 'card.py', 'ship', False, True,
       'a full model card from config + KL JSON + build record + sizes; TODO blocks for human prose, refuses a KL fingerprint mismatch or mixed builds'),
    _C('card-tables', 'vqlab.ship.card_tables', 'card_tables.py', 'ship', False, True,
       "a model card's KL table and paired deltas as Markdown, from kl-ladder JSON + each rung's own headers"),
    _C('slice', 'vqlab.build.slice_teacher', 'slice_teacher.py', 'build', False, True,
       'a small REAL slice of any teacher (--layers 0-3: those layers + embeddings/norm/head, config adjusted) for preflighting a writer in minutes'),
    _C('size', 'vqlab.ship.size', 'size.py', 'ship', False, True,
       "an artifact's size three ways (text / +tower / +MTP) and the full download, from its headers"),
    _C('teacher-prep', 'vqlab.build.teacher_prep', 'teacher_prep.py', 'build', False, True,
       'official release -> exact MLX teacher: cache guard, F8_E8M0 relabel, disk preflight, sanitize-stream'),
    _C('graft-extras', 'vqlab.build.graft_extras', 'graft_extras.py', 'build', False, True,
       "byte-copy a release's extra tensors (vision tower, image tokens, image-routing biases) into a text VQ artifact under the runtime's names; identity-probed"),
    _C('splice-ple', 'vqlab.build.splice_ple', 'splice_ple.py', 'build', False, True,
       'splice VQ PLE codes into a packed artifact'),
    _C('pack-dense', 'vqlab.build.pack_dense', 'pack_dense.py', 'build', False, True,
       'pack a dense VQ artifact'),
    _C('unpack-dense', 'vqlab.build.unpack_dense', 'unpack_dense.py', 'build', False, True,
       'diagnostic twin with PLAIN codes: isolates PACKING from every other variable (F141)'),
    _C('build-dense', 'vqlab.build.build_dense', 'build_dense.py', 'build', False, True,
       'splice dense VQ fits into a quantized base -> runnable artifact'),
    _C('reskeleton', 'vqlab.build.reskeleton', 'reskeleton.py', 'build', False, True,
       "put a VQ build's expert tensors on another build's skeleton (matched-skeleton comparisons)"),
    _C('graft', 'vqlab.build.graft_vision', 'graft_vision.py', 'build', False, True,
       'graft the bf16 vision tower into an artifact'),
    _C('ple-swap', 'vqlab.build.ple_swap', 'ple_swap.py', 'build', False, True,
       "swap an artifact's PLE tables for another rung's (symlinks; prices PLE bytes on KL)"),
    _C('mtp-extract', 'vqlab.build.mtp_extract', 'mtp_extract.py', 'build', False, True,
       "pull a model's MTP head out of its source checkpoint into a graft"),
    _C('mtp-head-ds4', 'vqlab.fit.mtp_head_ds4', 'mtp_head_ds4.py', 'build', False, True,
       'DeepSeek-V4-Flash draft head for a VQ trunk: exact mtp.0 via the runtime sanitize, VQ experts at trunk geometry, affine dense (the shipped q6 recipe)'),
    _C('mtp-pack', 'vqlab.build.mtp_pack', 'mtp_pack.py', 'build', False, True,
       'pack a bf16 MTP graft into a quantized drafting sidecar'),
    _C('mtp-graft', 'vqlab.build.mtp_graft', 'mtp_graft.py', 'build', False, True,
       "emit the MTP head in a native runtime's layout (language_model.mtp.*)"),
    _C('bundle', 'vqlab.bundle.add_model_file', 'add_model_file.py', 'bundle', False, True,
       '(re)write the self-contained model.py bundle (MoE)'),
    _C('patch-arch', 'vqlab.bundle.patch_arch', 'patch_arch.py', 'bundle', False, True,
       'surgical loader-aware arch fix on a published bundle, runtime untouched'),
    _C('vision-layout', 'vqlab.bundle.vision_layout', 'vision_layout.py', 'bundle', False, True,
       "check/repair a grafted tower's patch-embed memory layout"),
    _C('rebundle-dense', 'vqlab.bundle.rebundle_dense', 'rebundle_dense.py', 'bundle', False, True,
       "re-splice a DENSE artifact's model.py from the current runtime, in place"),
    _C('vision-smoke', 'vqlab.gate.vision_smoke', 'vision_smoke.py', 'gate', False, True,
       'vision arm of III.11: put an IMAGE through the shipping runtime'),
    _C('check', 'vqlab.gate.check_all', 'check_all.py', 'gate', True, True,
       'run the release gates that need no source model'),
    _C('smoke', 'vqlab.gate.smoke', 'smoke.py', 'gate', True, True,
       'generate one token through the runtime the artifact ships'),
    _C('spelling', 'vqlab.gate.spelling', 'spelling.py', 'gate', False, True,
       'fail on British spellings in released text (paper, cards); --fix rewrites to US'),
    _C('parity', 'vqlab.gate.parity', 'parity.py', 'gate', False, True,
       "a family's checklist of REFERENCE inference behaviours (clamps, eps, routing) and the test for each; exits 1 if a required one is untested"),
    _C('pin', 'vqlab.gate.pin', 'pin.py', 'gate', False, True,
       'freeze an artifact for measurement (symlinked weights, optional --runtime re-bake), smoke it, write vqlab_pin.json'),
    _C('verify', 'vqlab.gate.verify_artifact', 'verify_artifact.py', 'gate', True, True,
       'outlier gate: decode artifact bytes vs bf16 source'),
    _C('validate', 'vqlab.queue.validate', 'queue/validate.py', 'gate', True, True,
       'overnight validation queue: drain artifacts through the gates, never publish'),
    _C('check-release', 'vqlab.gate.check_release', 'check_release.py', 'gate', True, True,
       'release gate: files exist and function'),
    _C('check-bundle', 'vqlab.gate.check_bundle', 'check_bundle.py', 'gate', True, False,
       'bundle gate: shipped runtime matches repo runtime'),
    _C('check-comparator', 'vqlab.gate.check_comparator', 'check_comparator.py', 'gate', False, True,
       'comparator gate: tensor-set parity vs teacher'),
    _C('bundle-accept', 'vqlab.gate.bundle_accept', 'bundle_accept.py', 'gate', False, True,
       'kernel acceptance on the runtime lifted FROM the artifact'),
    _C('selftest', 'vqlab.selftest', 'selftest/__init__.py', 'gate', True, True,
       'run the real pipeline on a tiny synthetic model'),
    _C('mtp-smoke-head', 'vqlab.gate.mtp_smoke_head', 'mtp_smoke_head.py', 'gate', False, True,
       'head-alone load + T=1/T=2 forward timing (no trunk)'),
    _C('tasks', 'vqlab.score.score_tasks_streaming', 'score_tasks_streaming.py', 'score', True, True,
       'task benchmarks (HellaSwag/PIQA/WinoGrande via lm-eval), layer-streamed: scores models larger than RAM'),
    _C('score', 'vqlab.score.referee.score_streaming', 'referee/score_streaming.py', 'score', True, True,
       'streaming referee perplexity (models may exceed RAM)'),
    _C('stream-score', 'vqlab.score.stream_score', 'stream_score.py', 'score', True, True,
       "layer-streamed ppl / teacher top-k cache / KL-to-teacher: the KL gate's instrument (kl-ladder runs it per rung)"),
    _C('kl', 'vqlab.score.kl_damage', 'kl_damage.py', 'score', True, True,
       'KL-to-bf16 damage vs a cached teacher (cache/score)'),
    _C('kernel-truth', 'vqlab.score.kernel_truth', 'kernel_truth.py', 'score', False, True,
       'fused vs fallback: which is closer to an EXACT float64 reference? (F146)'),
    _C('kernel-truth-moe', 'vqlab.score.kernel_truth_moe', 'kernel_truth_moe.py', 'score', False, True,
       'MoE twin of kernel-truth: decode (small-N fused) vs prefill (gemmseg2) expert kernels against an exact float64 reference'),
    _C('act-stats', 'vqlab.score.act_stats', 'act_stats.py', 'score', False, True,
       "how often each clamp/limit in a family's parity checklist fires on the house corpora (GPU, streamed teacher)"),
    _C('kl-pair', 'vqlab.score.kl_pair', 'kl_pair.py', 'score', True, True,
       'paired KL between two arms scored in SEPARATE runs (env-var arms)'),
    _C('runtime-equiv', 'vqlab.score.runtime_equiv', 'runtime_equiv.py', 'score', False, True,
       'same model slice through two interpreters: bitwise equal, or max |logit diff| (F194; run whenever an env moves)'),
    _C('kl-ladder', 'vqlab.score.kl_ladder', 'kl_ladder.py', 'score', True, True,
       'rank rungs by KL against per-corpus teacher caches, with error bars'),
    _C('speed-pair', 'vqlab.bench.speed_pair', 'speed_pair.py', 'bench', False, True,
       'decode/prefill speed of TWO artifacts as a per-pair ratio (fresh process per arm, alternating, n>=3)'),
    _C('speed-pair-knurlogic', 'vqlab.bench.speed_pair_knurlogic', 'speed_pair_knurlogic.py', 'bench', False, True,
       'speed-pair for models Knurlogic serves (one Mac or a pipeline split): arms by model NAME, refuses identity collapse / non-bundled runtime'),
    _C('host-attrib', 'vqlab.bench.host_attrib', 'host_attrib.py', 'bench', False, True,
       "profile the VQ module's HOST cost (F41's unattributed 12-13%)"),
    _C('prefill-bench', 'vqlab.bench.prefill_bench', 'prefill_bench.py', 'bench', False, True,
       'time prefill on one artifact under BOTH codebook arms (ratio, n>=3)'),
    _C('active-bytes', 'vqlab.bench.active_bytes', 'active_bytes.py', 'bench', True, False,
       'weight bytes read per decode token, by component (the roofline denominator)'),
    _C('coverage', 'vqlab.bench.kernel_coverage', 'kernel_coverage.py', 'bench', True, True,
       'per-module report: which prefill kernel path each geometry takes'),
    _C('hc-micro', 'vqlab.bench.hc_micro', 'hc_micro.py', 'bench', False, True,
       'single GatedResidual micro-bench: is mx.compile inert on this chain? (split half invalid, see F133)'),
    _C('decode-timeline', 'vqlab.bench.decode_timeline', 'decode_timeline.py', 'bench', True, True,
       'every stage of one decode token, in order, timed, summing to the whole'),
    _C('stage-bandwidth', 'vqlab.bench.stage_bandwidth', 'stage_bandwidth.py', 'bench', True, False,
       'achieved GB/s and time above the roofline for EVERY decode stage (decode-timeline JSON + active-bytes billing; no GPU)'),
    _C('gpu-capture', 'vqlab.bench.gpu_capture', 'gpu_capture.py', 'bench', False, True,
       "record ONE decode step's Metal work as an Xcode .gputrace (per-kernel counters: occupancy, bandwidth, limiters); needs MTL_CAPTURE_ENABLED=1"),
    _C('serve-timeline', 'vqlab.bench.serve_timeline', 'serve_timeline.py', 'bench', True, False,
       'where a request SERVED by Knurlogic spends its time: its own partition (http, queue, admission, prefill/decode steps), n requests, median per bucket'),
    _C('prefill-timeline', 'vqlab.bench.prefill_timeline', 'prefill_timeline.py', 'bench', False, True,
       'every stage of one prefill (attention / MLP half of each layer), timed, summing to the whole; any arch'),
    _C('lut-bench', 'vqlab.bench.lut_bench', 'lut_bench.py', 'bench', False, True,
       'decode expert kernel with the codebook LUT read rewritten (free / rank-r / outer / two-level residual) vs stock: R3-5/R3-6 falsifier, arm per process'),
    _C('gdn-chunk-truth', 'vqlab.bench.gdn_chunk_truth', 'gdn_chunk_truth.py', 'bench', False, True,
       'chunked vs sequential gated-delta scan: error vs an fp64 reference + scan microbench (R3-3)'),
    _C('decode-ladder', 'vqlab.bench.decode_ladder', 'decode_ladder.py', 'bench', False, True,
       'per-component decode deletion arms (GDN / attn / VQ), Flash-aware'),
    _C('mtp-accept', 'vqlab.bench.mtp_accept', 'mtp_accept.py', 'bench', False, True,
       'paired draft-acceptance across prompts (the reliable instrument)'),
    _C('mtp-bench', 'vqlab.bench.mtp_bench', 'mtp_bench.py', 'bench', False, True,
       'measure the MTP speedup, acceptance and numerics control'),
    _C('mtp-arms', 'vqlab.bench.mtp_arms', 'mtp_arms.py', 'build', False, True,
       'one pinned, servable arm per draft head (symlinks; the release is never touched) for a head A/B via speed-pair-knurlogic --draft'),
    _C('manifest', 'vqlab.records.artifact_manifest', 'artifact_manifest.py', 'lab', True, False,
       'tamper stamp: were these shard bytes rewritten? (outside the artifact)'),
    _C('runs', 'vqlab.records.runlog', 'runlog.py', 'lab', True, False,
       'the run log: every vqlab invocation, argv, commit, exit code'),
    _C('provenance', 'vqlab.records.provenance', 'provenance.py', 'lab', True, False,
       'build record: how an artifact was made (tool, fitter, inputs, runtime, lineage)'),
    _C('scratch', 'vqlab.agents.scratch', 'scratch.py', 'lab', False, True,
       'scratch reclaimable: unpacked dirs whose verified pack exists, sizes + total, and the rm lines for a human (never deletes)'),
    _C('registry', 'vqlab.records.registry', 'registry.py', 'lab', True, False,
       'index of every artifact from provable facts; Hub drift check (metadata only)'),
    _C('publish', 'vqlab.ship.publish', 'publish.py', 'ship', False, True,
       'upload to the Hub, gated on check-release passing'),
    _C('mtp-generate', 'vqlab.ship.mtp_run', 'mtp_run.py', 'ship', False, True,
       'generate with MTP speculative drafting'),
    _C('serve', 'vqlab.ship.serve', 'serve.py', 'ship', False, True,
       'serve an artifact over an OpenAI-compatible API (mlx-lm server + MTP decode)'),
    _C('mcp', 'vqlab.agents.mcp_server', 'mcp_server.py', 'lab', False, True,
       'serve the lab to agents over MCP (stdio); one server per box'),
    _C('gui', 'vqlab.agents.gui', 'gui.py', 'lab', False, True,
       'a local, read-only window onto the lab (127.0.0.1:8781)'),
    _C('sz-pack', 'vqlab.skipzero.sz_pack', 'skipzero/sz_pack.py', 'build', False, True,
       'vq-skipzero: drop fully-dead VQ rows on disk (live rows + row mask), expanded at load; --dry-run prices it'),
    _C('sz-check', 'vqlab.skipzero.sz_check', 'skipzero/sz_check.py', 'gate', False, True,
       'numpy check that a vq-skipzero artifact expands to the original (live rows byte-identical, dead rows zero); --selftest'),
    _C('sz-resident', 'vqlab.skipzero.sz_resident_build', 'skipzero/sz_resident_build.py', 'build', False, True,
       'stage 2: stage-1 pack -> compact-RESIDENT artifact (row table, new dir)'),
    _C('sz-bitexact', 'vqlab.skipzero.sz_bitexact', 'skipzero/sz_bitexact.py', 'gate', False, True,
       'byte-equality gate, resident vs reference (modules N=1/8/4096/4097, logits, 9k prefill, greedy); --synthetic, --selftest'),
    _C('queue', 'vqlab.queue.runner', 'queue/runner.py', 'lab', False, True,
       'run a list of steps from PINNED code (git worktree), under the GPU lease, failing loudly; --preflight'),
    _C('reserve', 'vqlab.queue.reserve', 'queue/reserve.py', 'lab', False, True,
       'say who has a box until when (`reserve m4 --for X --until +3h`, --list, --release); queue run refuses a box reserved for someone else'),
)}

# Namespaces, in pipeline order: namespace -> (what it is for, {sub: flat}).
# A sub drops the namespace's own prefix (fit-moe -> fit moe) and otherwise
# keeps the flat name, so nothing has to be relearned twice. Every command
# sits in exactly one namespace (tests/test_commands.py).
NAMESPACES: dict[str, tuple[str, dict[str, str]]] = {
    'plan': ('what will it cost, where should the bits go, what IS this model', {
        'onboard': 'onboard',
        'family-profile': 'family-profile',
        'price': 'price',
        'zero-groups': 'zero-groups',
        'layer-leverage': 'layer-leverage',
        'loo-bands': 'loo-bands',
        'alloc-sweep': 'alloc-sweep',
        'probe-init': 'probe-init',
        'preflight-ram': 'preflight-ram',
        'preflight-disk': 'preflight-disk',
        'mtp-probe': 'mtp-probe',
        'mtp-probe35': 'mtp-probe35',
    }),
    'fit': ('teacher weights -> codebooks + codes', {
        'fits': 'fits',
        'moe': 'fit-moe',
        'dense': 'fit-dense',
        'ple': 'fit-ple',
        'additive': 'fit-additive',
        'geo-build': 'geo-build',
        'reselect': 'reselect',
        'harvest-parts': 'harvest-parts',
    }),
    'build': ('codes -> an artifact that loads (the build/ stage)', {
        'pack': 'pack',
        'pack-dense': 'pack-dense',
        'pack-ple': 'pack-ple',
        'splice-ple': 'splice-ple',
        'ple-swap': 'ple-swap',
        'unpack-dense': 'unpack-dense',
        'dense': 'build-dense',
        'reskeleton': 'reskeleton',
        'slice': 'slice',
        'minibase': 'minibase',
        'mix': 'mix',
        'stream-convert': 'stream-convert',
        'sanitize-stream': 'sanitize-stream',
        'teacher-prep': 'teacher-prep',
        'graft': 'graft',
        'graft-extras': 'graft-extras',
        'mtp-extract': 'mtp-extract',
        'mtp-pack': 'mtp-pack',
        'mtp-graft': 'mtp-graft',
        'mtp-head-ds4': 'mtp-head-ds4',
        'mtp-arms': 'mtp-arms',
        'sz-pack': 'sz-pack',
        'sz-resident': 'sz-resident',
    }),
    'bundle': ('ship the runtime inside the artifact', {
        'moe': 'bundle',
        'dense': 'rebundle-dense',
        'patch-arch': 'patch-arch',
        'vision-layout': 'vision-layout',
    }),
    'gate': ('is it loadable, correct and releasable', {
        'check': 'check',
        'check-release': 'check-release',
        'check-bundle': 'check-bundle',
        'check-comparator': 'check-comparator',
        'bundle-accept': 'bundle-accept',
        'verify': 'verify',
        'smoke': 'smoke',
        'vision-smoke': 'vision-smoke',
        'parity': 'parity',
        'selftest': 'selftest',
        'validate': 'validate',
        'pin': 'pin',
        'mtp-smoke-head': 'mtp-smoke-head',
        'spelling': 'spelling',
        'sz-check': 'sz-check',
        'sz-bitexact': 'sz-bitexact',
    }),
    'score': ('how much damage does it carry', {
        'kl-ladder': 'kl-ladder',
        'kl-pair': 'kl-pair',
        'kl': 'kl',
        'ppl': 'score',
        'stream-score': 'stream-score',
        'tasks': 'tasks',
        'kernel-truth': 'kernel-truth',
        'kernel-truth-moe': 'kernel-truth-moe',
        'act-stats': 'act-stats',
        'runtime-equiv': 'runtime-equiv',
    }),
    'bench': ('how fast is it, and where does the time go', {
        'decode-timeline': 'decode-timeline',
        'prefill-timeline': 'prefill-timeline',
        'decode-ladder': 'decode-ladder',
        'gdn-chunk-truth': 'gdn-chunk-truth',
        'lut-bench': 'lut-bench',
        'active-bytes': 'active-bytes',
        'stage-bandwidth': 'stage-bandwidth',
        'serve-timeline': 'serve-timeline',
        'gpu-capture': 'gpu-capture',
        'prefill-bench': 'prefill-bench',
        'coverage': 'coverage',
        'host-attrib': 'host-attrib',
        'hc-micro': 'hc-micro',
        'mtp-bench': 'mtp-bench',
        'mtp-accept': 'mtp-accept',
        'speed-pair': 'speed-pair',
        'speed-pair-knurlogic': 'speed-pair-knurlogic',
    }),
    'ship': ('publish or serve it', {
        'release-prep': 'release-prep',
        'size': 'size',
        'card': 'card',
        'card-tables': 'card-tables',
        'publish': 'publish',
        'serve': 'serve',
        'mtp-generate': 'mtp-generate',
    }),
    'lab': ('the lab around the pipeline: records, queues, boxes, agents', {
        'doctor': 'doctor',
        'config': 'config',
        'runs': 'runs',
        'provenance': 'provenance',
        'manifest': 'manifest',
        'registry': 'registry',
        'scratch': 'scratch',
        'queue': 'queue',
        'reserve': 'reserve',
        'mcp': 'mcp',
        'gui': 'gui',
    }),
}

# flat name -> its namespaced spelling, for help and error text
SPELLING = {flat: f"{ns} {sub}" for ns, (_, subs) in NAMESPACES.items()
            for sub, flat in subs.items()}


def resolve(argv):
    """(flat command, rest) for a namespaced or flat argv; (None, argv) if
    argv names neither. A namespace that is ALSO a flat command (`score`,
    `bundle`) is namespaced only when its next word is one of its subs:
    `vqlab score --corpus x` is still the flat referee scorer."""
    if not argv:
        return None, argv
    head = argv[0]
    if head in NAMESPACES and len(argv) > 1 and argv[1] in NAMESPACES[head][1]:
        return NAMESPACES[head][1][argv[1]], argv[2:]
    if head in COMMANDS:
        return head, argv[1:]
    return None, argv


def _code_spans(text):
    """re.findall(r"`([^`]+)`", text), without importing re (this module
    imports nothing: tests/test_commands.py)."""
    i = 0
    while True:
        a = text.find("`", i)
        b = text.find("`", a + 1) if a >= 0 else -1
        if b < 0:
            return
        if b == a + 1:          # an empty span: the regex retries one later
            i = a + 1
            continue
        yield text[a + 1:b]
        i = b + 1


def routed(text):
    """Flat names of the commands a Markdown text routes to: every `code`
    span read as a command line, namespaced (`fit moe ...`) or flat
    (`fit-moe ...`), with or without a leading `vqlab`. The selftest's
    CONTEXT.md check and its test share this, so both spellings count."""
    out = set()
    for span in _code_spans(text):
        words = span.split()
        if words and words[0] == "vqlab":
            words = words[1:]
        flat, _ = resolve(words)
        if flat:
            out.add(flat)
    return out


# A tool's own self-check that moved to vqlab.selftest (untangle step 14).
# The tool cannot import it (selftest sits above every stage), so the CLI
# runs this module instead when the command's argv carries the flag:
# {flat command: (flag or leading subcommand, module)}.
SELFTEST_ROUTES = {
    "reselect": ("selftest", "vqlab.selftest.reselect_fixture"),
    "sz-check": ("--selftest", "vqlab.selftest.sz_check_fixture"),
    "sz-bitexact": ("--selftest", "vqlab.selftest.sz_bitexact_cpu"),
}


def selftest_route(cmd, rest):
    """The module to run for `vqlab <cmd> <rest>` when rest asks for a moved
    self-check (a flag anywhere, a subcommand first), else None."""
    flag, module = SELFTEST_ROUTES.get(cmd, (None, None))
    if flag is None or any(a in ("-h", "--help") for a in rest):
        return None
    hit = flag in rest if flag.startswith("-") else rest[:1] == [flag]
    return module if hit else None


# Resumable builds: on a failure (GPU timeouts under disk contention) the MCP
# supervisor and the queue runner retry them DEFAULT_RETRIES times.
RESUMABLE = frozenset({"fit-moe", "fit-dense", "geo-build", "alloc-sweep", "validate"})
DEFAULT_RETRIES = 2


def mcp_allowlist() -> set[str]:
    """Commands an agent may launch over MCP `run`."""
    return {c.name for c in COMMANDS.values() if c.mcp}


def mcp_gpu_free() -> set[str]:
    """MCP-runnable commands that never touch the GPU (no lease, no exo gate)."""
    return {c.name for c in COMMANDS.values() if c.mcp and not c.gpu}
