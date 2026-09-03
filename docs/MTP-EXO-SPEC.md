# MTP drafting in exo — integration spec (draft for Noah, 2026-09-02)

Goal: the measured 1.56–1.80x (median 1.65x) single-box MTP speedup,
running on the exo cluster, for Flash-Next first and then the 397B and
GLM-5.3. Nothing here is implemented yet.

## Why this is a port, not a research project

- The head, its quantized sidecar (`mtp-head-q6.safetensors`, one file
  per family serving all rungs), and the draft/verify loop all exist in
  VQLab (`src/vqlab/mtp_head.py`, `vqlab mtp-pack` / `mtp-generate`),
  measured end-to-end on the same mlx-lm 0.31.9 runtime exo uses.
- exo's decode is literally `mlx_lm.generate.stream_generate`
  (generate.py:882, `prefill_step_size=1`) on the same qwen4_exp trunk
  classes `mtp-generate` drafts against. The capture point and loop
  transplant; they don't need re-derivation.

## Where it lands in exo

Replace the `stream_generate` call in `mlx_generate` with a
`speculative_stream_generate` used when a head is available, keeping the
plain path as the fallback (head absent, or `EXO_NO_MTP=1`).

1. **Head discovery/loading** (worker, load time): if the resolved model
   dir contains `mtp-head-*.safetensors`, the LAST pipeline rank loads it
   via `MTPHead.from_sidecar` (2.1 GiB resident; single-box rank counts
   as last). Other ranks load nothing. Placement must account for the
   +2.1 GiB on that rank (model card `mtp_head_size` field or stat the
   sidecar at placement time — the GLM rungs already reserve 2.9 GiB by
   card convention).
2. **Capture point** (all topologies): the head needs the trunk's final
   hidden state pre-lm_head. On the last rank the trunk runs the final
   norm + lm_head; port the mtp_decode.py capture (wrap/patch the
   lm_head call to stash `h`) into exo's model wrapper at load time,
   gated to the last rank. Zero cost when no head is loaded.
3. **Draft/verify loop** (replaces the per-token loop):
   - Last rank: after committing token t with hidden h_t, run
     `head.draft_logits(h_t, t+1)` → draft d1 (depth-1 to start; the
     397B's seq2/seq1=1.49 makes depth-2 a later, separate decision).
   - Verification: one trunk forward with T=2 (`[t+1, d1]`) through the
     normal pipeline path — all ranks participate exactly as they do for
     prefill chunks; a seq=2 step costs ~the same as seq=1 (measured
     49 vs 61 ms single-box — verification is better than free) and
     amortizes the TCP hop across 2 tokens when accepted.
   - Accept/reject per the mtp-generate loop; a rejected draft still
     yields the trunk's own token from the same forward (no wasted pass).
4. **Token distribution**: in today's exo pipeline every rank must see
   the committed tokens to advance its shard. Today that syncs one token
   per step; the loop changes this to 1–2 tokens per step. Reuse the
   existing pipeline send/recv machinery (same path prefill chunks use);
   the ranks stay in lockstep because verification IS a normal forward.
   This is the one place exo internals genuinely change; everything else
   is additive.
5. **Cache discipline**: the head keeps its own KV cache advanced one row
   per COMMITTED token (mtp_head.py handles the T>1 mask correctly —
   documented trap). On prefix-cache restore, the head cache must be
   rebuilt or seeded via `head.advance` over the restored region — or
   (simpler v1) dropped and re-seeded lazily; measure before optimizing.
   Trunk-side rollback is untouched: rejected drafts never enter the
   trunk cache (verification commits only accepted tokens).

## Rollout order

| stage | target | why |
|---|---|---|
| 0 | Flash-Next 2.1bpw, single box via exo | head+sidecar exist; isolates exo-integration bugs from cluster bugs; direct A/B vs `vqlab mtp-generate` numbers |
| 1 | Flash-Next, 2-node pipeline | proves the verify-through-pipeline path + token distribution |
| 2 | 397B | extract + quantize its `mtp.*` graft (weights exist upstream; our qwen4_exp port sanitizes them away — recover via `vqlab mtp-pack` against the bf16 source). Depth-1 buys little (seq2/seq1=1.49); measure before judging |
| 3 | GLM-5.3 | head = full `layers.45` MoE block (eh_proj/enorm/hnorm + 288 experts, ~13.5 GiB bf16 → ~2.9 GiB quantized, already reserved in rung sizes). Needs a GLM-shaped MTPHead subclass (mlx_vlm trunk classes, deltanet-adjacent block) — most new code of the three |

## Success criteria / honesty rules

- Stage 0 must reproduce ≥1.5x median vs the same box's non-MTP exo
  decode before any cluster work starts.
- Cluster claims quote measured tok/s on THIS pair over TCP, never the
  single-box numbers.
- Acceptance is prompt-dependent (0.578–0.812 measured); cards quote the
  range, not the max. Warm both paths before timing (the Metal-compile
  warmup trap is documented in the flash-next ledger).

## Open questions for Noah

1. Stage 0 needs ~50 GiB free on one box for the 2.1bpw + head — fine to
   run on the M4 alongside your GLM testing, or schedule it?
2. `EXO_NO_MTP` opt-out default-on drafting when a sidecar is present —
   or opt-in via card flag for the first release?
