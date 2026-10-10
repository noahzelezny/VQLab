"""Old module names, and the locators that find a module as a FILE.

Every module has ONE canonical dotted name: its path under src/vqlab
(`vqlab.runtime.vq_switch`, `vqlab.fit.fit_moe`). This file answers for the
other spellings, from two explicit tables -- nothing is globbed, so every
name it resolves is greppable here:

  PERMANENT   names shipped text depends on, kept forever:
                * every runtime/*.py stem, as `vqlab.<stem>` AND bare
                  `<stem>`: the SHIPPED runtime text (vq_dense.py's
                  _resolve_kernel, dense_shim.py) and published model.py
                  files name them, and inside a bundle the files sit flat
                  beside model.py. Must stay byte-identical, so the names
                  must stay too;
                * the public pre-split names: `vqlab.mtp_head*`,
                  `vqlab.mcp_server`, and `vq_397b_codes` / `pack_artifact`
                  (queue files and ledgers name them).
  DEPRECATED  every other old spelling (pre-split `vqlab.<stem>`, bare tool
              stems, modules moved or renamed in the 2026-10 untangle,
              `vqlab.assemble.*`). Importing one emits a DeprecationWarning
              naming the canonical module. Kept for one release.

Whatever name is used, the result is the SAME module object (_Alias): two
copies of vq_switch would mean two kernel caches and two sets of flag
globals. To add an alias, add a line to a table; there is no other source.

Locators, for code that reads a module as a FILE rather than importing it:

    find("smoke.py")               -> path of any tool (old filenames too)
    runtime_file("vq_switch.py")   -> the repo runtime a bundle is spliced from
    corpus("prose")                -> a house referee corpus
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import pathlib
import sys
import warnings

PKG = pathlib.Path(__file__).resolve().parent
SRC = PKG.parent

PERMANENT = {
    'arch_resolve': 'vqlab.runtime.arch_resolve',
    'dense_shim': 'vqlab.runtime.dense_shim',
    'gated_delta_chunked': 'vqlab.runtime.gated_delta_chunked',
    'glm5_shim': 'vqlab.runtime.glm5_shim',
    'pack_artifact': 'vqlab.build.pack',
    'runtime_profile': 'vqlab.runtime.runtime_profile',
    'vq_397b_codes': 'vqlab.fit.fit_moe',
    'vq_dense': 'vqlab.runtime.vq_dense',
    'vq_pack': 'vqlab.runtime.vq_pack',
    'vq_switch': 'vqlab.runtime.vq_switch',
    'vqlab.arch_resolve': 'vqlab.runtime.arch_resolve',
    'vqlab.build.pack_artifact': 'vqlab.build.pack',
    'vqlab.dense_shim': 'vqlab.runtime.dense_shim',
    'vqlab.fit.vq_397b_codes': 'vqlab.fit.fit_moe',
    'vqlab.gated_delta_chunked': 'vqlab.runtime.gated_delta_chunked',
    'vqlab.glm5_shim': 'vqlab.runtime.glm5_shim',
    'vqlab.mcp_server': 'vqlab.agents.mcp_server',
    'vqlab.mtp_head': 'vqlab.mtp.mtp_head',
    'vqlab.mtp_head_ds4': 'vqlab.build.mtp_head_ds4',
    'vqlab.mtp_head_glm5': 'vqlab.mtp.mtp_head_glm5',
    'vqlab.mtp_head_qwen35': 'vqlab.mtp.mtp_head_qwen35',
    'vqlab.pack_artifact': 'vqlab.build.pack',
    'vqlab.runtime_profile': 'vqlab.runtime.runtime_profile',
    'vqlab.vq_397b_codes': 'vqlab.fit.fit_moe',
    'vqlab.vq_dense': 'vqlab.runtime.vq_dense',
    'vqlab.vq_pack': 'vqlab.runtime.vq_pack',
    'vqlab.vq_switch': 'vqlab.runtime.vq_switch',
}

DEPRECATED = {
    'act_stats': 'vqlab.score.act_stats',
    'active_bytes': 'vqlab.bench.active_bytes',
    'add_model_file': 'vqlab.bundle.add_model_file',
    'additive_codebooks': 'vqlab.fit.additive_codebooks',
    'additive_vq': 'vqlab.fit.fit_additive',
    'alloc_sweep': 'vqlab.plan.alloc_sweep',
    'arch_finder': 'vqlab.core.arch_finder',
    'artifact': 'vqlab.core.artifact',
    'artifact_manifest': 'vqlab.records.artifact_manifest',
    'artifact_size': 'vqlab.core.artifact_size',
    'box_quiet': 'vqlab.bench.box_quiet',
    'build_dense': 'vqlab.build.build_dense',
    'build_dense_vq': 'vqlab.build.build_dense',
    'bundle_accept': 'vqlab.gate.bundle_accept',
    'card': 'vqlab.ship.card',
    'card_checks': 'vqlab.ship.card_checks',
    'card_tables': 'vqlab.ship.card_tables',
    'check_all': 'vqlab.gate.check_all',
    'check_bundle': 'vqlab.gate.check_bundle',
    'check_comparator': 'vqlab.gate.check_comparator',
    'check_release': 'vqlab.gate.check_release',
    'clamps': 'vqlab.core.clamps',
    'codepack': 'vqlab.core.codepack',
    'config_cmd': 'vqlab.agents.config_cmd',
    'decode_ladder': 'vqlab.bench.decode_ladder',
    'decode_timeline': 'vqlab.bench.decode_timeline',
    'doctor': 'vqlab.agents.doctor',
    'e134_accept': 'vqlab.gate.kernel_accept',
    'exo': 'vqlab.core.exo',
    'expert_src': 'vqlab.core.expert_src',
    'families': 'vqlab.family.registry',
    'family_profile': 'vqlab.plan.family_profile',
    'fit_additive': 'vqlab.fit.fit_additive',
    'fit_dense_vq': 'vqlab.fit.fit_dense_vq',
    'fit_moe': 'vqlab.fit.fit_moe',
    'fit_ple': 'vqlab.fit.fit_ple',
    'fits': 'vqlab.fit.fits',
    'fitstore': 'vqlab.core.fitstore',
    'gdn_chunk_truth': 'vqlab.bench.gdn_chunk_truth',
    'geo_build': 'vqlab.fit.geo_build',
    'gpu_capture': 'vqlab.bench.gpu_capture',
    'gpu_lease': 'vqlab.core.gpu_lease',
    'graft_extras': 'vqlab.build.graft_extras',
    'graft_vision': 'vqlab.build.graft_vision',
    'gui': 'vqlab.agents.gui',
    'harvest_parts': 'vqlab.fit.harvest_parts',
    'hc_micro': 'vqlab.bench.hc_micro',
    'host_attrib': 'vqlab.bench.host_attrib',
    'hub': 'vqlab.core.hub',
    'init_sweep': 'vqlab.plan.init_sweep',
    'kernel_accept': 'vqlab.gate.kernel_accept',
    'kernel_coverage': 'vqlab.bench.kernel_coverage',
    'kernel_reference': 'vqlab.gate.kernel_reference',
    'kernel_truth': 'vqlab.score.kernel_truth',
    'kernel_truth_moe': 'vqlab.score.kernel_truth_moe',
    'kl_damage': 'vqlab.score.kl_damage',
    'kl_ladder': 'vqlab.score.kl_ladder',
    'kl_pair': 'vqlab.score.kl_pair',
    'kmeans': 'vqlab.fit.kmeans',
    'knurlogic_client': 'vqlab.core.knurlogic_client',
    'layer_leverage': 'vqlab.plan.layer_leverage',
    'loo_bands': 'vqlab.build.loo_bands',
    'lut_bench': 'vqlab.bench.lut_bench',
    'mcp_server': 'vqlab.agents.mcp_server',
    'mem_budget': 'vqlab.core.mem_budget',
    'minibase': 'vqlab.build.minibase',
    'mix': 'vqlab.build.mix',
    'mtp_accept': 'vqlab.bench.mtp_accept',
    'mtp_arms': 'vqlab.bench.mtp_arms',
    'mtp_bench': 'vqlab.bench.mtp_bench',
    'mtp_extract': 'vqlab.build.mtp_extract',
    'mtp_graft': 'vqlab.build.mtp_graft',
    'mtp_head_ds4': 'vqlab.build.mtp_head_ds4',
    'mtp_pack': 'vqlab.build.mtp_pack',
    'mtp_probe': 'vqlab.plan.mtp_probe_qwen4_exp',
    'mtp_probe35': 'vqlab.plan.mtp_probe_qwen35',
    'mtp_probe_qwen35': 'vqlab.plan.mtp_probe_qwen35',
    'mtp_probe_qwen4_exp': 'vqlab.plan.mtp_probe_qwen4_exp',
    'mtp_run': 'vqlab.ship.mtp_run',
    'mtp_smoke_head': 'vqlab.gate.mtp_smoke_head',
    'numerics': 'vqlab.family.version_stamp',
    'onboard': 'vqlab.agents.onboard',
    'pack': 'vqlab.build.pack',
    'pack_dense': 'vqlab.build.pack_dense',
    'pack_ple': 'vqlab.build.pack_ple',
    'parity': 'vqlab.gate.parity',
    'patch_arch': 'vqlab.bundle.patch_arch',
    'pin': 'vqlab.gate.pin',
    'pin_record': 'vqlab.core.pin_record',
    'ple_stream': 'vqlab.core.ple_stream',
    'ple_swap': 'vqlab.build.ple_swap',
    'power': 'vqlab.bench.power',
    'prefill_bench': 'vqlab.bench.prefill_bench',
    'prefill_timeline': 'vqlab.bench.prefill_timeline',
    'preflight_disk': 'vqlab.plan.preflight_disk',
    'preflight_ram': 'vqlab.plan.preflight_ram',
    'price': 'vqlab.plan.price',
    'probe_init_sweep': 'vqlab.plan.init_sweep',
    'provenance': 'vqlab.records.provenance',
    'publish': 'vqlab.ship.publish',
    'queue_eta': 'vqlab.queue.eta',
    'rebundle_dense': 'vqlab.bundle.rebundle_dense',
    'registry': 'vqlab.records.registry',
    'release_prep': 'vqlab.ship.release_prep',
    'reselect': 'vqlab.fit.reselect',
    'reserve': 'vqlab.queue.reserve',
    'reskeleton': 'vqlab.build.reskeleton',
    'run_queue': 'vqlab.agents.run_queue',
    'runlog': 'vqlab.records.runlog',
    'runtime_equiv': 'vqlab.score.runtime_equiv',
    'runtime_load': 'vqlab.family.model_loader',
    'sanitize_stream': 'vqlab.build.sanitize_stream',
    'score_tasks_streaming': 'vqlab.score.score_tasks_streaming',
    'scratch': 'vqlab.agents.scratch',
    'scratch_cmd': 'vqlab.agents.scratch',
    'selftest': 'vqlab.selftest',
    'serve': 'vqlab.ship.serve',
    'serve_timeline': 'vqlab.bench.serve_timeline',
    'settings': 'vqlab.agents.settings',
    'size': 'vqlab.ship.size',
    'size_cmd': 'vqlab.ship.size',
    'slice_teacher': 'vqlab.build.slice_teacher',
    'smoke': 'vqlab.gate.smoke',
    'speed_pair': 'vqlab.bench.speed_pair',
    'speed_pair_knurlogic': 'vqlab.bench.speed_pair_knurlogic',
    'spelling': 'vqlab.gate.spelling',
    'splice_ple': 'vqlab.build.splice_ple',
    'stage_bandwidth': 'vqlab.bench.stage_bandwidth',
    'stamp_fields': 'vqlab.core.stamp_fields',
    'step_verdict': 'vqlab.records.step_verdict',
    'stream_convert': 'vqlab.build.stream_convert',
    'stream_score': 'vqlab.score.stream_score',
    'teacher_prep': 'vqlab.build.teacher_prep',
    'timeline_common': 'vqlab.bench.timeline_common',
    'unpack_dense': 'vqlab.build.unpack_dense',
    'us_spelling': 'vqlab.gate.spelling',
    'validate_queue': 'vqlab.queue.validate',
    'verify_artifact': 'vqlab.gate.verify_artifact',
    'verify_dev_codebook': 'vqlab.gate.verify_dev_codebook',
    'verify_e134_devcb': 'vqlab.gate.verify_dev_codebook',
    'version_stamp': 'vqlab.family.version_stamp',
    'vision_layout': 'vqlab.bundle.vision_layout',
    'vision_smoke': 'vqlab.gate.vision_smoke',
    'vqlab.act_stats': 'vqlab.score.act_stats',
    'vqlab.active_bytes': 'vqlab.bench.active_bytes',
    'vqlab.add_model_file': 'vqlab.bundle.add_model_file',
    'vqlab.additive_codebooks': 'vqlab.fit.additive_codebooks',
    'vqlab.additive_vq': 'vqlab.fit.fit_additive',
    'vqlab.agents.queue_eta': 'vqlab.queue.eta',
    'vqlab.agents.reserve': 'vqlab.queue.reserve',
    'vqlab.agents.scratch_cmd': 'vqlab.agents.scratch',
    'vqlab.alloc_sweep': 'vqlab.plan.alloc_sweep',
    'vqlab.arch_finder': 'vqlab.core.arch_finder',
    'vqlab.artifact': 'vqlab.core.artifact',
    'vqlab.artifact_manifest': 'vqlab.records.artifact_manifest',
    'vqlab.artifact_size': 'vqlab.core.artifact_size',
    'vqlab.assemble': 'vqlab.build',
    'vqlab.assemble.build_dense': 'vqlab.build.build_dense',
    'vqlab.assemble.build_dense_vq': 'vqlab.build.build_dense',
    'vqlab.assemble.codepack': 'vqlab.core.codepack',
    'vqlab.assemble.graft_extras': 'vqlab.build.graft_extras',
    'vqlab.assemble.graft_vision': 'vqlab.build.graft_vision',
    'vqlab.assemble.loo_bands': 'vqlab.build.loo_bands',
    'vqlab.assemble.minibase': 'vqlab.build.minibase',
    'vqlab.assemble.mix': 'vqlab.build.mix',
    'vqlab.assemble.mtp_extract': 'vqlab.build.mtp_extract',
    'vqlab.assemble.mtp_graft': 'vqlab.build.mtp_graft',
    'vqlab.assemble.mtp_head_ds4': 'vqlab.build.mtp_head_ds4',
    'vqlab.assemble.mtp_pack': 'vqlab.build.mtp_pack',
    'vqlab.assemble.pack': 'vqlab.build.pack',
    'vqlab.assemble.pack_artifact': 'vqlab.build.pack',
    'vqlab.assemble.pack_dense': 'vqlab.build.pack_dense',
    'vqlab.assemble.pack_ple': 'vqlab.build.pack_ple',
    'vqlab.assemble.ple_swap': 'vqlab.build.ple_swap',
    'vqlab.assemble.reskeleton': 'vqlab.build.reskeleton',
    'vqlab.assemble.sanitize_stream': 'vqlab.build.sanitize_stream',
    'vqlab.assemble.slice_teacher': 'vqlab.build.slice_teacher',
    'vqlab.assemble.splice_ple': 'vqlab.build.splice_ple',
    'vqlab.assemble.stream_convert': 'vqlab.build.stream_convert',
    'vqlab.assemble.teacher_prep': 'vqlab.build.teacher_prep',
    'vqlab.assemble.unpack_dense': 'vqlab.build.unpack_dense',
    'vqlab.box_quiet': 'vqlab.bench.box_quiet',
    'vqlab.build.build_dense_vq': 'vqlab.build.build_dense',
    'vqlab.build_dense': 'vqlab.build.build_dense',
    'vqlab.build_dense_vq': 'vqlab.build.build_dense',
    'vqlab.bundle_accept': 'vqlab.gate.bundle_accept',
    'vqlab.card': 'vqlab.ship.card',
    'vqlab.card_checks': 'vqlab.ship.card_checks',
    'vqlab.card_tables': 'vqlab.ship.card_tables',
    'vqlab.check_all': 'vqlab.gate.check_all',
    'vqlab.check_bundle': 'vqlab.gate.check_bundle',
    'vqlab.check_comparator': 'vqlab.gate.check_comparator',
    'vqlab.check_release': 'vqlab.gate.check_release',
    'vqlab.clamps': 'vqlab.core.clamps',
    'vqlab.codepack': 'vqlab.core.codepack',
    'vqlab.config_cmd': 'vqlab.agents.config_cmd',
    'vqlab.core.families': 'vqlab.family.registry',
    'vqlab.core.numerics': 'vqlab.family.version_stamp',
    'vqlab.core.runtime_load': 'vqlab.family.model_loader',
    'vqlab.core.version_stamp': 'vqlab.family.version_stamp',
    'vqlab.decode_ladder': 'vqlab.bench.decode_ladder',
    'vqlab.decode_timeline': 'vqlab.bench.decode_timeline',
    'vqlab.doctor': 'vqlab.agents.doctor',
    'vqlab.e134_accept': 'vqlab.gate.kernel_accept',
    'vqlab.exo': 'vqlab.core.exo',
    'vqlab.expert_src': 'vqlab.core.expert_src',
    'vqlab.families': 'vqlab.family.registry',
    'vqlab.family.arch': 'vqlab.core.arch_finder',
    'vqlab.family_profile': 'vqlab.plan.family_profile',
    'vqlab.fit.additive_vq': 'vqlab.fit.fit_additive',
    'vqlab.fit.mtp_head_ds4': 'vqlab.build.mtp_head_ds4',
    'vqlab.fit_additive': 'vqlab.fit.fit_additive',
    'vqlab.fit_dense_vq': 'vqlab.fit.fit_dense_vq',
    'vqlab.fit_moe': 'vqlab.fit.fit_moe',
    'vqlab.fit_ple': 'vqlab.fit.fit_ple',
    'vqlab.fits': 'vqlab.fit.fits',
    'vqlab.fitstore': 'vqlab.core.fitstore',
    'vqlab.gate.e134_accept': 'vqlab.gate.kernel_accept',
    'vqlab.gate.selftest': 'vqlab.selftest',
    'vqlab.gate.us_spelling': 'vqlab.gate.spelling',
    'vqlab.gate.validate_queue': 'vqlab.queue.validate',
    'vqlab.gate.verify_e134_devcb': 'vqlab.gate.verify_dev_codebook',
    'vqlab.gdn_chunk_truth': 'vqlab.bench.gdn_chunk_truth',
    'vqlab.geo_build': 'vqlab.fit.geo_build',
    'vqlab.gpu_capture': 'vqlab.bench.gpu_capture',
    'vqlab.gpu_lease': 'vqlab.core.gpu_lease',
    'vqlab.graft_extras': 'vqlab.build.graft_extras',
    'vqlab.graft_vision': 'vqlab.build.graft_vision',
    'vqlab.gui': 'vqlab.agents.gui',
    'vqlab.harvest_parts': 'vqlab.fit.harvest_parts',
    'vqlab.hc_micro': 'vqlab.bench.hc_micro',
    'vqlab.host_attrib': 'vqlab.bench.host_attrib',
    'vqlab.hub': 'vqlab.core.hub',
    'vqlab.init_sweep': 'vqlab.plan.init_sweep',
    'vqlab.kernel_accept': 'vqlab.gate.kernel_accept',
    'vqlab.kernel_coverage': 'vqlab.bench.kernel_coverage',
    'vqlab.kernel_reference': 'vqlab.gate.kernel_reference',
    'vqlab.kernel_truth': 'vqlab.score.kernel_truth',
    'vqlab.kernel_truth_moe': 'vqlab.score.kernel_truth_moe',
    'vqlab.kl_damage': 'vqlab.score.kl_damage',
    'vqlab.kl_ladder': 'vqlab.score.kl_ladder',
    'vqlab.kl_pair': 'vqlab.score.kl_pair',
    'vqlab.kmeans': 'vqlab.fit.kmeans',
    'vqlab.knurlogic_client': 'vqlab.core.knurlogic_client',
    'vqlab.layer_leverage': 'vqlab.plan.layer_leverage',
    'vqlab.loo_bands': 'vqlab.build.loo_bands',
    'vqlab.lut_bench': 'vqlab.bench.lut_bench',
    'vqlab.mem_budget': 'vqlab.core.mem_budget',
    'vqlab.minibase': 'vqlab.build.minibase',
    'vqlab.mix': 'vqlab.build.mix',
    'vqlab.mtp_accept': 'vqlab.bench.mtp_accept',
    'vqlab.mtp_arms': 'vqlab.bench.mtp_arms',
    'vqlab.mtp_bench': 'vqlab.bench.mtp_bench',
    'vqlab.mtp_extract': 'vqlab.build.mtp_extract',
    'vqlab.mtp_graft': 'vqlab.build.mtp_graft',
    'vqlab.mtp_pack': 'vqlab.build.mtp_pack',
    'vqlab.mtp_probe': 'vqlab.plan.mtp_probe_qwen4_exp',
    'vqlab.mtp_probe35': 'vqlab.plan.mtp_probe_qwen35',
    'vqlab.mtp_probe_qwen35': 'vqlab.plan.mtp_probe_qwen35',
    'vqlab.mtp_probe_qwen4_exp': 'vqlab.plan.mtp_probe_qwen4_exp',
    'vqlab.mtp_run': 'vqlab.ship.mtp_run',
    'vqlab.mtp_smoke_head': 'vqlab.gate.mtp_smoke_head',
    'vqlab.numerics': 'vqlab.family.version_stamp',
    'vqlab.onboard': 'vqlab.agents.onboard',
    'vqlab.pack': 'vqlab.build.pack',
    'vqlab.pack_dense': 'vqlab.build.pack_dense',
    'vqlab.pack_ple': 'vqlab.build.pack_ple',
    'vqlab.parity': 'vqlab.gate.parity',
    'vqlab.patch_arch': 'vqlab.bundle.patch_arch',
    'vqlab.pin': 'vqlab.gate.pin',
    'vqlab.pin_record': 'vqlab.core.pin_record',
    'vqlab.plan.mtp_probe': 'vqlab.plan.mtp_probe_qwen4_exp',
    'vqlab.plan.mtp_probe35': 'vqlab.plan.mtp_probe_qwen35',
    'vqlab.plan.onboard': 'vqlab.agents.onboard',
    'vqlab.plan.probe_init_sweep': 'vqlab.plan.init_sweep',
    'vqlab.ple_stream': 'vqlab.core.ple_stream',
    'vqlab.ple_swap': 'vqlab.build.ple_swap',
    'vqlab.power': 'vqlab.bench.power',
    'vqlab.prefill_bench': 'vqlab.bench.prefill_bench',
    'vqlab.prefill_timeline': 'vqlab.bench.prefill_timeline',
    'vqlab.preflight_disk': 'vqlab.plan.preflight_disk',
    'vqlab.preflight_ram': 'vqlab.plan.preflight_ram',
    'vqlab.price': 'vqlab.plan.price',
    'vqlab.probe_init_sweep': 'vqlab.plan.init_sweep',
    'vqlab.provenance': 'vqlab.records.provenance',
    'vqlab.publish': 'vqlab.ship.publish',
    'vqlab.queue_eta': 'vqlab.queue.eta',
    'vqlab.rebundle_dense': 'vqlab.bundle.rebundle_dense',
    'vqlab.records.card': 'vqlab.ship.card',
    'vqlab.records.card_tables': 'vqlab.ship.card_tables',
    'vqlab.records.scratch_cmd': 'vqlab.agents.scratch',
    'vqlab.records.size_cmd': 'vqlab.ship.size',
    'vqlab.registry': 'vqlab.records.registry',
    'vqlab.release_prep': 'vqlab.ship.release_prep',
    'vqlab.reselect': 'vqlab.fit.reselect',
    'vqlab.reserve': 'vqlab.queue.reserve',
    'vqlab.reskeleton': 'vqlab.build.reskeleton',
    'vqlab.run_queue': 'vqlab.agents.run_queue',
    'vqlab.runlog': 'vqlab.records.runlog',
    'vqlab.runtime_equiv': 'vqlab.score.runtime_equiv',
    'vqlab.runtime_load': 'vqlab.family.model_loader',
    'vqlab.sanitize_stream': 'vqlab.build.sanitize_stream',
    'vqlab.score_tasks_streaming': 'vqlab.score.score_tasks_streaming',
    'vqlab.scratch': 'vqlab.agents.scratch',
    'vqlab.scratch_cmd': 'vqlab.agents.scratch',
    'vqlab.serve': 'vqlab.ship.serve',
    'vqlab.serve_timeline': 'vqlab.bench.serve_timeline',
    'vqlab.settings': 'vqlab.agents.settings',
    'vqlab.ship.size_cmd': 'vqlab.ship.size',
    'vqlab.size': 'vqlab.ship.size',
    'vqlab.size_cmd': 'vqlab.ship.size',
    'vqlab.slice_teacher': 'vqlab.build.slice_teacher',
    'vqlab.smoke': 'vqlab.gate.smoke',
    'vqlab.speed_pair': 'vqlab.bench.speed_pair',
    'vqlab.speed_pair_knurlogic': 'vqlab.bench.speed_pair_knurlogic',
    'vqlab.spelling': 'vqlab.gate.spelling',
    'vqlab.splice_ple': 'vqlab.build.splice_ple',
    'vqlab.stage_bandwidth': 'vqlab.bench.stage_bandwidth',
    'vqlab.stamp_fields': 'vqlab.core.stamp_fields',
    'vqlab.step_verdict': 'vqlab.records.step_verdict',
    'vqlab.stream_convert': 'vqlab.build.stream_convert',
    'vqlab.stream_score': 'vqlab.score.stream_score',
    'vqlab.teacher_prep': 'vqlab.build.teacher_prep',
    'vqlab.timeline_common': 'vqlab.bench.timeline_common',
    'vqlab.unpack_dense': 'vqlab.build.unpack_dense',
    'vqlab.us_spelling': 'vqlab.gate.spelling',
    'vqlab.validate_queue': 'vqlab.queue.validate',
    'vqlab.verify_artifact': 'vqlab.gate.verify_artifact',
    'vqlab.verify_dev_codebook': 'vqlab.gate.verify_dev_codebook',
    'vqlab.verify_e134_devcb': 'vqlab.gate.verify_dev_codebook',
    'vqlab.version_stamp': 'vqlab.family.version_stamp',
    'vqlab.vision_layout': 'vqlab.bundle.vision_layout',
    'vqlab.vision_smoke': 'vqlab.gate.vision_smoke',
    'vqlab.zero_groups': 'vqlab.plan.zero_groups',
    'zero_groups': 'vqlab.plan.zero_groups',
}

assert not {"queue", "vqlab.queue"} & (set(PERMANENT) | set(DEPRECATED))   # stdlib `queue` stays the stdlib's
assert not set(PERMANENT) & set(DEPRECATED)


def _names():
    """{importable alias: canonical dotted name}, both tables."""
    return {**PERMANENT, **DEPRECATED}


# Folders find() searches, in order (it locates FILES, by old filename too).
_FIND_DIRS = ("", "runtime", "core", "family", "plan", "fit", "build", "bundle",
              "gate", "score", "bench", "records", "ship", "agents", "mtp",
              "queue", "selftest", "skipzero", "ops", "util", "score/referee")


def find(name: str) -> pathlib.Path:
    """Path of a module FILE by its filename or relative path. Old filenames
    (in old queue files and shell chains) resolve through the alias tables."""
    stem, dot, ext = name.rpartition("/")[2].partition(".")
    canon = _names().get(stem)
    if canon and ext == "py":
        new = canon.rsplit(".", 1)[1]
        name = name[:len(name) - len(stem + dot + ext)] + new + dot + ext
    for d in _FIND_DIRS:
        p = PKG / d / name
        if p.exists():
            return p
    if name.startswith("referee/"):                 # old relative spelling
        return find(name.split("/", 1)[1])
    raise FileNotFoundError(f"vqlab: no module file {name!r} under {PKG}")


def runtime_file(name: str) -> pathlib.Path:
    """The repo's copy of a shipped runtime file (vq_switch.py, vq_dense.py,
    vq_pack.py, dense_shim.py, glm5_shim.py). This is the text bundlers
    splice and check-bundle compares against; its flag DEFAULTS are the
    repo's, so a caller that needs a specific profile applies it with
    runtime_profile.apply_profile / resolve_runtime."""
    p = PKG / "runtime" / name
    if not p.exists():
        raise FileNotFoundError(f"vqlab: {name} is not a runtime file ({p})")
    return p


CORPORA = {"prose": "referee_corpus.txt",
           "code": "referee_corpus_code_public.txt",
           "lit": "referee_corpus_literary.txt"}
_CORPUS_ALIAS = {"wikitext": "prose", "code-public": "code", "literary": "lit"}


def corpus(name: str = "prose") -> pathlib.Path:
    """A house referee corpus by NAME (prose / code / lit), never by a path
    built from __file__: a scratch script that hardcodes the repo layout
    breaks silently when the layout moves (night 4, 2026-09-26). A path
    argument is passed through legacy_path, so old spellings still resolve."""
    key = _CORPUS_ALIAS.get(name, name)
    if key in CORPORA:
        p = PKG / "score" / "referee" / CORPORA[key]
    else:
        p = pathlib.Path(legacy_path(str(name)))
    if not p.is_file():
        raise FileNotFoundError(f"vqlab: no corpus {name!r} (have {sorted(CORPORA)})")
    return p


def legacy_path(p):
    """Map a path recorded BEFORE the stage split onto its new home.

    Teacher caches store their corpus path in meta.json (absolute or
    repo-relative) and shell chains pass corpora by path; both still name
    src/vqlab/referee/. Existing paths are returned unchanged."""
    s = str(p)
    if pathlib.Path(s).exists():
        return p
    # "vqlab/referee/" (no src/) also covers an INSTALLED package's site-packages path
    for old, new in (("vqlab/referee/", "vqlab/score/referee/"),):
        if old in s:
            cand = s.replace(old, new)
            if pathlib.Path(cand).exists() or not pathlib.Path(s).is_absolute():
                return type(p)(cand) if not isinstance(p, str) else cand
    return p


class _Alias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """ONE module object per module, whatever name it is imported by.

    Without this, `import vq_switch` and `vqlab.runtime.vq_switch` load the
    file twice: two kernel caches, two sets of env-flag globals, and a test
    that reloads one name silently leaves the other untouched. Sits FIRST on
    sys.meta_path so it wins over the ordinary path search -- including a
    script's own directory, which Python puts on sys.path[0]. It only
    answers for names in the two tables, so nothing else is affected.
    """

    def __init__(self):
        self.map = _names()
        self._fresh = set()     # ids of modules create_module just imported

    def _canon(self, fullname):
        return self.map.get(fullname)

    def find_spec(self, fullname, path=None, target=None):
        new = self._canon(fullname)
        if new is None or new == fullname:
            return None
        if fullname in DEPRECATED:
            warnings.warn(f"{fullname} is deprecated; import {new}",
                          DeprecationWarning, stacklevel=2)
        t = importlib.util.find_spec(new)
        return importlib.util.spec_from_loader(
            fullname, self, origin=t.origin if t else f"alias of {new}")

    def create_module(self, spec):
        mod = importlib.import_module(self._canon(spec.name))
        sys.modules[spec.name] = mod
        self._fresh.add(id(mod))
        return mod

    def exec_module(self, module):
        """First import: the canonical import already executed it. RELOAD
        (importlib.reload by the alias name, which is how the flag tests
        re-read env defaults) must genuinely re-execute. Either way the
        module keeps its CANONICAL identity; the import system just stamped
        the alias spec on it."""
        name = module.__spec__.name
        canon = self._canon(name) or name
        # Ask the path finder for the FILE's spec. importlib.util.find_spec
        # would return the module's current __spec__ -- the alias spec the
        # import system just stamped on it -- and recurse back into us.
        parent, _, _ = canon.rpartition(".")
        spec = importlib.machinery.PathFinder.find_spec(
            canon, importlib.import_module(parent).__path__)
        if id(module) in self._fresh:
            self._fresh.discard(id(module))
        else:
            spec.loader.exec_module(module)
        module.__spec__, module.__name__, module.__loader__ = (
            spec, spec.name, spec.loader)

    # `python -m vqlab.<old name>` goes through runpy, which asks the loader
    # for code rather than a module. Hand it the real module's code.
    def _target(self, fullname):
        return importlib.util.find_spec(self._canon(fullname))

    def get_code(self, fullname):
        t = self._target(fullname)
        return t.loader.get_code(t.name)

    def get_source(self, fullname):
        t = self._target(fullname)
        return t.loader.get_source(t.name)

    def is_package(self, fullname):
        return False


def install_import_aliases():
    if not any(isinstance(f, _Alias) for f in sys.meta_path):
        sys.meta_path.insert(0, _Alias())


install = install_import_aliases   # pre-step-15 name; one release
install_import_aliases()
