# gemma-4-e4b-it — VQ-PLE (7.39 GiB)

**The community 8-bit, with a better embedding table.** This is
mlx-community's `gemma-4-e4b-it-8bit` with exactly one change: the 5.25 GiB
per-layer-embedding table (35% of the model's bytes) is replaced by a
vector-quantized version at 5.75 bits/weight. Everything else — attention,
MLPs, norms, towers — is byte-identical to the artifact you already know.

That one swap makes the model **measurably closer to bf16 than the 8-bit
it came from**, at 1 GiB less disk and 1.8 GB less peak RAM.

## Measured results

All numbers on the same instruments, same corpus, same teacher cache;
"incumbent" = mlx-community gemma-4-e4b-it-8bit as shipped.

| | **this artifact** | incumbent 8-bit |
|---|---|---|
| size on disk | **7.39 GiB** | 8.38 GiB |
| KL to bf16 (literary corpus) | **7.451 mnats/token** | 8.149 mnats/token |
| top-1 agreement with bf16 | 95.70% | 95.70% |
| litbench (cyclic, generative, n=104) | 81.73% | 84.62% |
| — paired McNemar | 7 discordant items, 5–2, p=0.45 — statistically indistinguishable | |
| decode | 77.4 tok/s | 84.2 tok/s |
| prompt processing | 392 tok/s | 496 tok/s |
| peak memory (short chat) | **7.2 GB** | 9.0 GB |

Honest summary: closer to the bf16 teacher on the precise instrument (KL),
indistinguishable on the noisy one (litbench, n=104 cannot resolve a
3-point gap — SE ±3.7), ~8% slower decode, 20% less RAM. If you are
RAM-bound, this is a strict upgrade; if you are latency-bound, keep the
8-bit.

## Why the embedding table

We first vector-quantized everything (MLPs + embeddings) and LOST to the
8-bit decisively (20.8 vs 8.1 mnats): e4b's MLP weights do not tolerate
5.75-bit VQ even though their reconstruction error looks excellent — fit
error and output damage are different quantities. An ablation split the
damage: the MLP contributed essentially all of it, and the VQ embedding
table alone was *better* than its 8-bit affine counterpart. So this
artifact keeps the 8-bit MLPs and ships only the swap that wins.

Embeddings are the friendly case for VQ at runtime too: a lookup decodes
only the rows a batch touches — no matmul kernel, no full-table
materialization, which is where the RAM saving comes from.

## No calibration data

The VQ fit is pure weight-space k-means against the bf16 tensors: no
calibration corpus, no forward passes, no distillation. 154 seconds of
codebook fitting on an M3 Ultra. Every number above was measured after,
not optimized for.

## Run it

```bash
pip install mlx-lm
mlx_lm.chat --model <this artifact>
```

The artifact is self-contained (`model.py` ships inside it); stock mlx_lm
loads it with no extra code. It also loads STRICTLY — the upstream 8-bit
artifact ships 126 tensors for KV-shared layers that mlx_lm never
instantiates and silently drops; those are removed here.

## Verification

- VQ table verified decode-side against the bf16 source (relerr 0.0296,
  uniform across all 262,144 rows).
- Packed codes verified bit-exact against the unpacked reference, and the
  packed artifact reproduces the KL score to the third decimal.
- Corrupting the codebook garbles generation — the VQ path is provably
  live, not a fallback.

## Limitations

- ~8% slower decode and ~20% slower prefill than the 8-bit incumbent.
- litbench point estimate is 3 points below the incumbent; the paired test
  says noise (p=0.45), and the KL says closer-to-teacher, but if your use
  case resembles literary MC comprehension specifically, measure your own.
- Vision/audio towers are carried unchanged from the incumbent artifact;
  vision was not re-benched here.

### Multi-machine (exo) note

If you shard this across an exo cluster: VQ codebooks must **replicate,
not slice**. The guard is bundled in `model.py`; upstream fix is
[exo PR #2268](https://github.com/exo-explore/exo/pull/2268).
