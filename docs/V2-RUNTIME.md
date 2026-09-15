# The v2 runtime — what it is, where it lives, and how to ship it

**One-line status (2026-09-15): the v2 runtime is DONE, committed, and is the
repo default — but NOT YET PUBLISHED. The Hub still serves the arc6 runtime
frozen 2026-09-09. Shipping v2 = re-bundle + re-publish the 20 artifacts.
Publish is gated on Noah.**

This doc exists because "the v2 runtime" was only ever recorded as scattered
findings (F50–F58) and flag defaults inside `src/vqlab/vq_switch.py`; a fresh
agent asked to "ship v2" had nothing named to find. This is that named thing.

## What "v2" is

The v2 runtime is **`src/vqlab/vq_switch.py` at its current committed state**,
with the performance flags defaulting ON. It is a **speed** upgrade over the
published arc6 runtime, at **identical or 1-ULP-equivalent** numerics. It does
NOT change weights, codes, bpw, or file layout — only the bundled `model.py`
kernel code.

**Measured, same-session interleaved on 35B-3.4 vs the SHIPPED arc6 bundle
(F56 / F58):**
- **+11.9% prefill** (~2151 → ~2398 tok/s)
- **+3.3–3.8% decode**, plus the d4 decode walker's **+12% decode** on d4
  geometries (F58)
- Parity vs affine-8bit: 82% → **90.5% prefill**, ~93.6% decode (F60 addendum)

### The flag stack (all default ON in the committed runtime)

| flag (env override) | default | what it does | finding |
|---|---|---|---|
| `VQ_GEMMSEG_OTILE64` (`_GEMMSEG_OT2`) | ON | output-block pairing, +5–6% prefill | F54 |
| `VQ_GEMMSEG_PH2V` (`_GEMMSEG_PH2V`) | ON | vectorized phase-2 staging | F56 |
| `VQ_GEMMSEG_BF16IO` (`_GEMMSEG_BF16IO`) | ON | bf16 I/O, single-round | F51 |
| `VQ_DECODE_BF16IO` (`_DECODE_BF16IO`) | ON | bf16 decode I/O | F53 |
| `VQ_D4_WALK` (`_D4_WALK`) | ON | d4 decode bit-walker, +12% decode, bit-exact | F58 |
| routing memo + vectorized tile build | on (in code) | +1.9% prefill, bit-identical | F45 |

Off-by-default experiments that are NOT part of v2: `VQ_GEMMSEG_XT_PAD`,
`VQ_GEMMSEG_DSTORE`, `VQ_GEMMSEG_PIPE`. Leave them off.

### The numerics caveat (why it was parked)

Most of v2 is bit-identical; a few pieces (bf16-I/O single-round, the arc-4
simd_sum reduction) are **1-ULP-equivalent**, not bit-exact. Standing policy
is to **re-run PPL at any numerics-changing release**. That coupled v2 to a
future "v2 model release" — but v2 is weight-identical, so it can ship as a
**runtime-only refresh** ahead of any new-weights work, gated on a PPL
spot-check (see below).

## How publishing actually works (the mechanism)

exo loads a model via `trust_remote_code` / an unconditional `model_file`
execution — **the runtime IS the `model.py` bundled inside each artifact.**
There is no separately-installed package. So:

- **What's published** = each repo's `model.py`, frozen at bundle time
  (2026-09-09 for all 20 = arc6).
- **`bundle` re-splices the CURRENT `src/vqlab/vq_switch.py`** into an
  artifact's `model.py` (`add_model_file.py:83` reads vq_switch.py verbatim).
- Therefore **shipping v2 = re-bundle each artifact, then re-upload.** No
  weights move; only `model.py` changes.

## Ship-everywhere procedure (20 repos)

Per artifact (`../.venv/bin/python` from vqlab; HF token via `HF_HOME`):

```bash
# 1. re-bundle: splice current v2 vq_switch.py into the artifact's model.py
PYTHONPATH=src ../.venv/bin/python -m vqlab.cli bundle --artifact <ARTIFACT_DIR>

# 2. publish (runs check-release gate WITH a generation smoke first; refuses
#    on any gate failure). Small/mid rungs go single-box on the M3:
PYTHONPATH=src ../.venv/bin/python -m vqlab.cli publish --artifact <ARTIFACT_DIR>

# 3. big rungs that exceed one box (the 397B family, Flash-Next) use a live
#    2-node exo instance placed per rung and torn down:
#    publish --artifact <DIR> --cluster-smoke <URL> --cluster-peer user@host
```

Notes / gotchas (all incident-born):
- **README-only prose fixes** upload straight via `hf` and skip the smoke;
  the gate guards RUNTIME, not prose. But a runtime change (this) MUST re-gate.
- Big-rung loads MUST be node-local, never over SMB (45% after 30 min).
  Tar-over-ssh a local copy first (~7 min / 141 GB).
- The peer-hash check catches byte-different-but-semantically-equal config
  serializations — sync BYTES across nodes.
- Re-gate every rung; a bundle is one process + one streamed pass.

## PPL spot-check for the runtime-only ship

Because v2 is 1-ULP not strictly bit-exact, run PPL on **2 representative
rungs** (one d4 e.g. 35B-3.4, one d8 e.g. a Flash rung) old-model.py vs
new-model.py; if Δppl is within noise, the 1-ULP concern is retired for the
whole lineup (the kernels are geometry-shared, not per-rung). This is far
cheaper than the full v2-model-release PPL sweep and is the gate for a
runtime-only refresh.

## Relationship to the re-selection campaign (KL-TUNED-CODEBOOKS-PLAN.md)

Orthogonal. v2 is a **runtime/kernel** speed upgrade (same weights). Code
re-selection (F67–F98) changes **weights/codes** for quality at fixed bytes.
They ship independently:
- v2 runtime → re-bundle + re-publish existing 20 artifacts (this doc).
- re-selection → new artifact weights, its own quality gate, a v2-MODEL
  release.
If you want "v2 Flash," decide which you mean: the faster runtime (this,
ready now) or re-selected/re-allocated Flash weights (the campaign, still
being refereed). They can go in the same release or separately.
