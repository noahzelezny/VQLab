# Card update draft — bundle refresh + serving guide (2026-09-02)

For Noah's review. Nothing here is published. Sections marked [ALL] go on
every updated card; [FLASH]/[397B] are per-family. This draft folds in
the earlier docs/card-section-prefill-memory.md (long-prompt guidance),
which should ship in the same pass.

---

## [ALL] Recent changes (new section, near the top)

**2026-09 bundle refresh.** The bundled `model.py` runtime is updated:

- Faster MoE expert kernels: threadgroup occupancy fix, template-free
  specialized dispatch, device-direct activation reads, and a simd_sum
  reduction for the packed d8 geometry; uint32 code loads for d2.
  Everything except the reduction is **bit-identical** to the previous
  bundle (verified per kernel against real weights); the reduction is
  1-ULP equivalent — measured quality delta is exactly zero (identical
  perplexity to every digit, KL 0.0 between old and new logits), only
  the last-bit rounding of ties differs. Nothing about the model's
  quality numbers changes. Set VQ_D8_SS=0 to restore bit-identical
  legacy behavior.
- The bundle now loads under BOTH `mlx-lm` and `mlx_vlm` (nested-config
  coercion is scoped per runtime).
- **Nothing you have already downloaded breaks.** The previous revision
  remains fetchable (pin the prior commit hash if you need it), the
  weights are unchanged — this refresh replaces only `model.py` and
  `config.json` keys — and the old bundle keeps working on the runtimes
  it always worked on.

Measured effect (same box, same prompt, A-B-A):
[FLASH 2.1bpw] decode 17.4 -> 18.8 tok/s (+8%); with speculative
decoding (below) ~25 tok/s.
[397B] no measurable decode change (the cluster interconnect dominates);
the refresh is for runtime consistency across the lineup.
[35B / gemma-26b] not separately measured; same kernels, same
equivalence guarantee as above.

## [ALL] Run it with `vqlab serve` (new section, after "Run it")

The bundle needs no server of its own — any mlx-lm-compatible stack
works — but VQLab ships one with the extras wired up:

```bash
git clone https://github.com/noahzelezny/VQLab && cd VQLab
python3 -m venv .venv && . .venv/bin/activate && pip install -e .
python -m vqlab.cli serve --model TheDrainFlorist/<this-repo>
```

OpenAI-compatible API on localhost. Flags worth knowing:
- `--sidecar mtp-head-q6.safetensors` — speculative decoding, if this
  repo ships the sidecar (see below). 1.3-1.4x decode at identical
  outputs-distribution (the trunk verifies every drafted token).
- Long prompts: see "Long prompts and memory" below [from the
  card-section-prefill-memory.md draft — prefill chunk sizing,
  mx.set_memory_limit, per-chunk eval].

## [ALL MoE w/ MTP head] Speculative decoding (MTP) — SHIPPING (decision 2026-09-02)

This repo includes `mtp-head-q6.safetensors` (2.1 GiB): the model's own
multi-token-prediction head, quantized, packaged so that stock loaders
ignore it entirely (it costs nothing unless asked for by name).
Measured on this artifact (M3 Ultra, greedy, 378-token runs):
18.1 tok/s plain -> 24.4 tok/s drafting, acceptance 0.77, output
distribution preserved by exact rejection sampling. `vqlab serve
--sidecar` or `vqlab mtp-generate` enables it; nothing else changes.

DECIDED 2026-09-02: sidecars SHIP for every family with an MTP head —
Flash (4 repos), GLM (pending shim validation), 397B (pending head
build/probe; single-box serving via vqlab serve at day one, cluster MTP
under exo lands with pipeline-verify later). Stock loaders ignore the
sidecar file; nothing changes for users who don't opt in.

## [ALL] exo branch pointer correction (edit existing Hardware section)

`vq-codebook-replicate` -> `vq-serving` (same guard, plus the serving
fixes: per-chunk prefill eval, per-family chunk sizing, duck-typed cache
handling, optional MTP stage 0). One-line edit per card.

## [397B] Cluster numbers (updates the "not yet measured" line)

Replace "Cluster throughput: not yet measured" with measured values
(2-box M3 Ultra 96GB + M4 128GB, tensor sharding, 300-token greedy):
- TCP (Thunderbolt bridge, no RDMA): ~17 tok/s
- RDMA (TB5, jaccl): ~20.4 tok/s
[Pending tonight: honest same-harness comparison vs spicyneuron 3.5bit
and our 2.6bpw — add the table when those land, including the plain
statement that the affine build decodes faster and ours is smaller at
matched-or-better quality. We win on bytes, they win on tok/s; say so.]

## Pre-push checklist (internal, not card text)

- [x] publish --all exclude list (working files can no longer ship)
- [x] check-release PASS on re-bundled 2.1bpw (full gate incl. smoke) —
      re-run 2026-09-02 on the FINAL kernel: PASS, 8 tokens, strict
      resolution confirms `_fused`/`_dense_fused`/`VQSwitchLinear`/
      `VQPLEEmbedding` all resolve from the artifact's own model.py
- [x] static gates PASS: all 18 re-bundled artifacts (23 paths incl. the 5
      symlink aliases), check_release --no-smoke, 2026-09-02. 18/18 PASS.
- [x] referee spot-check on re-bundled 2.1bpw (5.9025 vs published 5.9003,
      attributed to traversal order; within cross-instrument floor)
- [x] referee spot-check, FINAL kernel, prefill chunk 16 so the decode
      kernel actually dispatches (old .pre-arc5 bundle vs new, two separate
      processes): nll 1.7778030182234943 / ppl 5.916842932532446 on BOTH
      arms — identical to all 16 digits, reproducing d4855e4 exactly.
- [x] exo serve-smoke: 397B 2.2/2.4/2.6 PASS; Flash 4.4 PASS (9.05 tok/s
      cold), Flash 5.5 PASS (6.96 tok/s cold) 2026-09-02 evening
- [x] spicyneuron 2.6bit vs VQ 2.6bpw RDMA table (29.9 vs 20.3 tok/s;
      affine reads 5.3% more bytes/token — we win bytes, they win tok/s)
- [x] Noah: sidecar decision — SHIP for all MTP-capable families
- [ ] GLM full-trunk shim validation (warm A/B running; acceptance 0.82
      confirmed on shipping artifact)
- [x] **GLM prefill: fully diagnosed, both ranks** (2026-09-02 night, after
      syncing M4's exo to M3's commit). Transient is FIXED on both ranks
      (M3 peak +2.4G, M4 peak +2.6G per chunk, flat over 26k tokens). The
      +50G Noah observed on the M4 = its ~70G shard + ~5G KV/state + ~35G
      MLX allocator/wired retention above active — box hit 0.1G free at
      minimum. Mitigation deployed: EXO_MLX_MEM_LIMIT_GB=82 in the M4's
      ~/.exo/exo-env.sh (hook added to its supervisor, mirroring M3's).
      Takes effect on next M4 exo restart; verify headroom on next GLM
      session. GLM un-gated for the push; card gets the honest memory-
      requirement sentence.
      26,423-token prompt through the 2-node pipeline: per-chunk transient
      <2G, flat across all 13 chunks (peak 50.5G -> 52.2G on a 48.8G-resident
      rank; the rise is KV growth). The morning's chunk-2048 + per-chunk
      eval + mem-limit fixes were the whole cure; per-layer eval not needed.
      vqlab MTP loop head-seeding also now chunked (O(chunk) not O(prompt),
      identical-tokens + identical-seed gates, 296 tests green). GLM is
      un-gated for the push pending normal card review.
- [ ] 397B head build + probe (agent running)
- [x] final-kernel confirm A-B-A + re-bundle + check-release canary (one
      consolidated pass, 2026-09-02 overnight). The count was x13 in this
      draft; the actual backup-marked set is **18** physical artifacts
      (the 5 extra `.pre-rows8` paths are SYMLINK aliases onto the same
      dirs). A-B-A on the 2.1bpw, one 44.96 GiB load, 300-token greedy,
      A,B,A,B,A,B:
        arm A (shipped rows-8 bundle) 18.124 / 18.091 / 18.023  med 18.091
        arm B (final devx+simd_sum)   19.029 / 19.285 / 19.085  med 19.085
        +5.49%; each arm self-consistent across its 3 runs.
      Greedy divergence at token 36 of 300 (' careful' -> ' detailed'),
      both continuations coherent — the sanctioned 1-ULP signature, and
      the referee scores zero delta on the same weights (above).
- [x] all 18 re-bundled onto the final kernel; every model.py carries
      `_SRC_FUSED_PACKED_D8_SIMD_DEVX_SS`, compiles, and check-bundle
      PASSes. config.json byte-unchanged on all 18 (the bundler's config
      rewrite is a no-op here). Backups: model.py.pre-arc5 +
      config.json.pre-arc5 next to each.
- [x] MTP sidecar staged into all 4 Flash-Next VQ dirs (2.1/3.2/4.4/5.5),
      2.140 GiB each, from the store-root master. GLM (6.094) and 397B
      (5.412) sidecars are DIFFERENT heads and were not touched — the
      three files share the name `mtp-head-q6.safetensors` but differ in
      geometry (hidden 2560 / 4096 / 4096); never cross-copy them.
- [ ] **MORNING REVIEW — two pre-existing bundle defects, found and fixed
      by this pass, that no gate had caught on these four artifacts**:
      the 3 dense 27B artifacts shipped a model.py carrying vq_dense.py
      but NOT vq_switch.py, and gemma-4-e4b-it-VQ-PLE carried NEITHER
      runtime (9,960 bytes: loader shim only) plus a literal
      `from mlx_lm.models.vq_...` import. All four therefore required a
      VQ-patched mlx-lm and would have raised ModuleNotFoundError on a
      stock install. They now carry both runtimes and pass check-bundle +
      check-release --no-smoke. Smoked after the repair: the three 27B
      artifacts PASS the full gate (8 tokens, runtime resolved from the
      artifact). gemma-4-e4b-it-VQ-PLE does NOT — see below.
- [ ] **gemma-4-e4b-it-VQ-PLE — NO-SHIP, new blocker, unrelated to the
      kernel work**: with the bundle repaired it now gets far enough to
      generate and dies in mlx's OWN attention kernel:
        [metal::Device] Unable to load kernel steel_attention_float32_
        bq32_bk16_bd256_... Threadgroup memory size (53760) exceeds the
        maximum threadgroup memory allowed (32768)
      bd256 = head_dim 256; 32768 is the M3 threadgroup limit. This is not
      VQ code and not the re-bundle. Confirmed pre-existing and previously
      INVISIBLE: the old bundle fails the static gate first (forbidden
      `from mlx_lm.models.vq_` at line 114), so it could never reach
      generation — the packaging defect was masking a runtime defect.
      Same 32 KiB threadgroup ceiling already noted for the mtp-probe35
      head. Likely M3-specific; NOT verified on the M4 (out of scope for
      this pass). Do not publish this artifact on the strength of a score.
- [ ] Noah: card text review
- [ ] Push = new revision per repo, old revision noted as pinnable
