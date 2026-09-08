# State — 2026-09-07 evening (d8 promoted; 397B cluster failure solved)

Supersedes nothing in STATE-2026-09-07.md; that document's open item #1
(the 397B-2.2 warmup failure) is now closed, and #2 (the d8 bench) is done.

## The 397B-2.2 failure: solved, and it was none of the earlier guesses

Not memory. Not the VQ kernel. Not MTP. It was a dtype bug at exo's
pipeline seam.

`PipelineFirstLayer` received with `mx.distributed.recv_like(x, ...)`,
which takes shape AND dtype from the LOCAL template `x`. On the 397B's
rank 1 that template was float32, so the whole last shard ran float32.
`head_dim=256` in float32 makes MLX's steel attention want 53760 B of
threadgroup memory against a 32768 B cap, so the kernel could not LOAD
and the runner aborted. Measured with an env-gated probe:

    rank 0: PipelineLastLayer  in x.dtype = bfloat16
    rank 1: PipelineFirstLayer pre-recv   = float32

Fix (exo 574a7bd7): take the recv dtype from a real parameter of the
rank's own first layer. 397B-2.2 has served since.

**Scope, corrected (exo 12038d1b).** The first commit claimed this hit
every pipeline-sharded model. It does not — that was a generalisation
from one measurement:

| model | family | rank-1 pre-recv |
|---|---|---|
| 397B-2.2 | qwen3_5_moe | **float32** — affected |
| Flash-4.4 | qwen4_exp | bfloat16 — fine |
| GLM-2.7 | glm5_next | bfloat16 — fine |

GLM/Flash cluster numbers stand. **Why the 397B differs is UNKNOWN** —
the obvious story (its rank-1 shard has no loaded embed_tokens to take a
template from) is a hypothesis, not a measurement.

Independently real but NOT the cause: the 397B's MTP sidecar stores seven
RMSNorm gains as F32 where the trunk uses BF16 (exo 85e1ec8e — flagged to
the MTP session, marked REVERT ME, and the load-time cast there is a
safety net that must not be mistaken for a fix). With EXO_MTP=0 the model
still died identically, which is what exonerated it.

Also fixed: a live rank desync. The M3 launcher dropped its
EXO_PREFILL_STEP_SIZE default on 09-02 and the M4 mirror did not, so
GLM-5.3 ran 2048 on rank 0 and 4096 on rank 1. `qwen3.5: 4096` is now
explicit in PREFILL_STEP_SIZE_BY_FAMILY (it is the measured value, not
the default leaking through), and the M4 default is removed in canon.
NOTE: `~/exo-supervised-m4.sh` is a DEPLOYED COPY, re-synced from
`scripts/exo-supervised-worker.sh` on every service start — editing it on
the M4 is silently reverted.

## d8 promoted to default (vqlab 1815885)

| artifact | placement | d8 fused | OFF | ON | gain |
|---|---|---|---|---|---|
| 397B-2.2 | ring | 151 (29->180/180) | 25.184 s | 19.489 s | 1.29x |
| Flash-2.1 | M3 single | 92 (6->98/144) | 15.190 s | 11.587 s | 1.31x |

Method caveat: the flag is import-time, so the arms are separate loads,
not interleaved like the d2/d4 benches. See D8-BENCH-2026-09-07.md.

Fleet is now 100% fused everywhere except Flash-2.1 (98/144): 46 of its
d8 modules have NSUB=80 and `gemmseg_fits` wants a multiple of 32. That
gap was invisible until d8 was armed. It is the top remaining kernel item.

## Cluster state as left

- Both nodes: `VQ_MOE_FUSED_GEMM_D8=1`, `EXO_MTP=0`, `EXO_DTYPE_PROBE=1`.
- GLM-2.7 is the currently placed instance.
- Flash-4.4 copied to the M4 and verified: 153 files, every name+size
  matching, 103,396,153,110 bytes both nodes. Serves 2-node in 44 s.
- Flash-2.1 and 3.2 are M3-only by design; 4.4 and 5.5 are on both.

## Open

1. **Rebundle/republish is NOT done and is deliberately gated.** Published
   artifacts carry their own `model.py` with `_FUSED_GEMM_D8` still "0", so
   the promotion changes nothing already shipped until Noah runs that step.
2. **MTP validation** waits on the rebuilt sidecar. Measuring acceptance
   against the F32-norm head would measure the defect; a low number would
   be a false negative.
3. **Why the 397B's rank-1 template is float32** — unprobed.
4. **NSUB=80 tail** on Flash-2.1 (46 modules).
5. **Strip `EXO_DTYPE_PROBE`** from auto_parallel.py once publishes are done.
6. Recurring operational papercut: deleting an instance regularly leaves
   runners wedged in `ShuttingDown` forever, and a placement issued while
   the ring is still syncing its event log is silently dropped. Only an
   exo restart clears the wedged runners. Placement needs a retry loop.
