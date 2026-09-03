# Card update draft — bundle refresh + serving guide (2026-09-02)

For Noah's review. Nothing here is published. Sections marked [ALL] go on
every updated card; [FLASH]/[397B] are per-family. This draft folds in
the earlier docs/card-section-prefill-memory.md (long-prompt guidance),
which should ship in the same pass.

---

## [ALL] Recent changes (new section, near the top)

**2026-09 bundle refresh.** The bundled `model.py` runtime is updated:

- Faster MoE expert kernels (threadgroup occupancy fix for the packed
  d8 geometry; uint32 code loads for d2). Outputs are **bit-identical**
  to the previous bundle — verified per kernel against real weights —
  so nothing about the model's quality numbers changes.
- The bundle now loads under BOTH `mlx-lm` and `mlx_vlm` (nested-config
  coercion is scoped per runtime).
- **Nothing you have already downloaded breaks.** The previous revision
  remains fetchable (pin the prior commit hash if you need it), the
  weights are unchanged — this refresh replaces only `model.py` and
  `config.json` keys — and the old bundle keeps working on the runtimes
  it always worked on.

Measured effect (same box, same prompt, A-B-A):
[FLASH 2.1bpw] decode 17.4 -> 18.1 tok/s (+3.9%); with speculative
decoding (below) 24.4 tok/s.
[397B] no measurable decode change (the cluster interconnect dominates);
the refresh is for runtime consistency across the lineup.
[35B / gemma-26b] not separately measured; same kernels, same
bit-identity guarantee.

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

## [FLASH] Speculative decoding (MTP) — OPTIONAL, only if we ship sidecars

This repo includes `mtp-head-q6.safetensors` (2.1 GiB): the model's own
multi-token-prediction head, quantized, packaged so that stock loaders
ignore it entirely (it costs nothing unless asked for by name).
Measured on this artifact (M3 Ultra, greedy, 378-token runs):
18.1 tok/s plain -> 24.4 tok/s drafting, acceptance 0.77, output
distribution preserved by exact rejection sampling. `vqlab serve
--sidecar` or `vqlab mtp-generate` enables it; nothing else changes.

DECISION NEEDED: shipping sidecars adds 2.1 GiB x 4 Flash repos and
makes the MTP claims public. If deferred, this whole section is dropped
and the sidecars stay local.

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
- [x] check-release PASS on re-bundled 2.1bpw (full gate incl. smoke)
- [x] static gates PASS: 35B x4, Flash 2.1/3.2, gemma-26b
- [x] referee spot-check on re-bundled 2.1bpw (5.9025 vs published 5.9003,
      attributed to traversal order; within cross-instrument floor)
- [x] exo serve-smoke: 397B 2.2/2.4/2.6 PASS; Flash 4.4 PASS (9.05 tok/s
      cold), Flash 5.5 PASS (6.96 tok/s cold) 2026-09-02 evening
- [x] spicyneuron 2.6bit vs VQ 2.6bpw RDMA table (29.9 vs 20.3 tok/s;
      affine reads 5.3% more bytes/token — we win bytes, they win tok/s)
- [ ] Noah: sidecar ship/defer decision
- [ ] Noah: card text review
- [ ] Push = new revision per repo, old revision noted as pinnable
