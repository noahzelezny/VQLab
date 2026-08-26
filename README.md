# VQLab

**Custom size-targeted vector-quantized (VQ) builds of large models on Apple
Silicon, with MLX.** Name a byte budget, price the recipe before fitting,
fit it data-free, pack to true bit-width, verify, and serve through stock
`mlx-lm` — the runtime ships inside the artifact.

The method and every number below are documented in the companion paper,
*Data-Free Vector Quantization Beats Affine Quantization at Matched Bytes*
(publication pending; link will land here). Released artifacts are published
under [TheDrainFlorist](https://huggingface.co/TheDrainFlorist) on Hugging
Face with their VQ runtimes bundled in-checkpoint (stock `mlx-lm`, no
patches).

## Published models

Every artifact this tool's pipeline has shipped:

| model | family |
|---|---|
| [Qwen3.5-397B-A17B-VQ-2.2bpw](https://huggingface.co/TheDrainFlorist/Qwen3.5-397B-A17B-VQ-2.2bpw) | 397B MoE (d8/K16384) |
| [Qwen3.5-397B-A17B-VQ-2.4bpw](https://huggingface.co/TheDrainFlorist/Qwen3.5-397B-A17B-VQ-2.4bpw) | 397B MoE |
| [Qwen3.5-397B-A17B-VQ-2.6bpw](https://huggingface.co/TheDrainFlorist/Qwen3.5-397B-A17B-VQ-2.6bpw) | 397B MoE |
| [Qwen3.5-397B-A17B-VQ-3bpw](https://huggingface.co/TheDrainFlorist/Qwen3.5-397B-A17B-VQ-3bpw) | 397B MoE (flagship) |
| [Qwen3.6-35B-A3B-VQ-3.4bpw](https://huggingface.co/TheDrainFlorist/Qwen3.6-35B-A3B-VQ-3.4bpw) | 35B MoE |
| [Qwen3.6-35B-A3B-VQ-3.8bpw](https://huggingface.co/TheDrainFlorist/Qwen3.6-35B-A3B-VQ-3.8bpw) | 35B MoE |
| [Qwen3.6-35B-A3B-VQ-4.6bpw](https://huggingface.co/TheDrainFlorist/Qwen3.6-35B-A3B-VQ-4.6bpw) | 35B MoE |
| [Qwen3.6-35B-A3B-VQ-5.4bpw](https://huggingface.co/TheDrainFlorist/Qwen3.6-35B-A3B-VQ-5.4bpw) | 35B MoE |
| [Qwen3.8-27B-VQ-3.9bpw](https://huggingface.co/TheDrainFlorist/Qwen3.8-27B-VQ-3.9bpw) | dense 27B |
| [Qwen3.8-27B-VQ-4.5bpw](https://huggingface.co/TheDrainFlorist/Qwen3.8-27B-VQ-4.5bpw) | dense 27B |
| [Qwen3.8-27B-VQ-4.8bpw](https://huggingface.co/TheDrainFlorist/Qwen3.8-27B-VQ-4.8bpw) | dense 27B |
| [gemma-4-26b-a4b-it-VQ-6.2bpw](https://huggingface.co/TheDrainFlorist/gemma-4-26b-a4b-it-VQ-6.2bpw) | gemma MoE * |
| [gemma-4-e4b-it-VQ-PLE](https://huggingface.co/TheDrainFlorist/gemma-4-e4b-it-VQ-PLE) | gemma dense * |

\* Released and usable, but no quality claims are made for the gemma family
anywhere in this repo or the paper: its scoring instrument is
non-deterministic (see "Known scope limits"). All sizes are measured packed bytes; every margin is stated against
a measured seed-noise floor; see [METHODOLOGY.md](METHODOLOGY.md) for the
rules that keep these numbers honest.

## Measured results

**397B MoE (Qwen3.5-397B-A17B), streaming-referee perplexity, prose/code:**

| build | GiB | prose ppl | code ppl |
|---|---|---|---|
| d8/K16384 | 100.97 | 3.0591 | 2.6728 |
| flat K256 (2.4bpw) | 111.62 | 2.7655 | 2.6383 |
| **flat K512** | **122.31** | **2.5634** | **2.6123** |
| flat K2048 (flagship) | 143.68 | 2.3410 | 2.5963 |
| calibrated 2.6-bit comparator | 120.6 | 3.1843 | 2.6667 |
| calibrated 3.5-bit comparator | 165.6 | 2.3614 | 2.6005 |

- **flat K512 beats the calibrated 2.6-bit build by 0.6209 prose ppl (24x the
  K256 floor) and by 0.0544 on code (3.1x), at +1.7 GiB** — the cleanest
  like-for-like on the ladder.
- **d8/K16384 beats the same comparator while being 19.6 GiB smaller**
  (prose margin 0.1252 = 4.9x the K256 floor). No d8 floor has been
  measured, so that multiple is *borrowed* and reads as a lower bound on
  confidence, not a measurement. It is the ~101 GiB build a 128 GB Mac holds.
- **The flagship is 21.9 GiB smaller than the calibrated 3.5-bit build**, with
  prose better by 0.0204 = 3.6x the K2048 floor (claimed) and code a tie
  (0.4x, inside the floor). "Wins both corpora" is withdrawn.

*Reproducibility note:* prose-ppl rows are fully reproducible from the
shipped corpus. The **code-ppl column was measured on a private corpus that
does not ship**; `scripts/make_code_corpus.py` builds a public replacement
with a provenance manifest, but that is a different instrument and its
scores do not compare to the column above.

**Read the floors with the margins.** Seed-noise floors are geometry-specific
and narrow as K grows: 397B d4/K256 = 0.0256 prose / ~0.0178 code; d4/K2048 =
0.0056 / 0.0104; dense 27B d2/K256 = 2.085 mnats / 0.0447 ppl. A margin
inside its floor is not a claim, and a floor borrowed from another geometry
is labelled as borrowed.

**Size basis, disclosed rather than restated.** Our 397B builds carry the
bf16 vision tower (0.8494 GiB); both calibrated comparators are text-only.
Every 397B size advantage above is therefore *understated* by roughly that
much. We keep the download-size convention and disclose the offset, because
restating sizes would move every number in our own favour.

**35B MoE (Qwen3.6-35B-A3B), KL-to-bf16 — quality only:** our d4/K8192
build measures **53.022 mnats KL / 89.55% top-1**, against the community
4-bit's 78.557 / 85.61% on the same instrument — a **32% KL reduction**.
The packed artifact passed the outlier gate on a second box, generated
through its shipping kernel, and reproduced its score to every printed
digit.

*No size comparison is claimed for this pair.* Our 35B rungs are text-only
while the community quants carry a 333-tensor bf16 vision tower
(0.832 GiB measured), so the two sizes are computed on mismatched bases
and the delta moves once restated. Our artifact is 14.838 GiB packed,
measured; the comparison to the incumbent's footprint is deferred rather
than quoted on a basis we know to be mismatched.

**Dense 27B (KL-to-bf16 + ppl):** the recipe is not an MoE phenomenon.
d2/K512 (the published artifact: 32.81 mnats / 90.84% top-1 / 5.162 ppl)
beats the 4-bit affine conversion by 28.4% KL (6.2x floor) and +1.02 pp
top-1 at 4-bit-class size; d4/K1024 beats q3 on both metrics at
0.35 GiB less.

**Where the method ends, measured:** the VQ advantage lives at the low-bpw
end — wins measured at 2.0–4.5 bpw across three families; the VQ/affine
crossover is bracketed at 4.5–6.0 bpw on the dense 27B; by 8 bits affine is
essentially lossless and the advantage is gone. Prefill remains ~0.5x affine
at 35B. We publish these fences as measured boundaries, not caveats.

## The differentiating feature: size targeting

Flat rungs leave gaps. VQLab prices an artifact **before fitting it**:

```bash
vqlab price --family qwen397b --budget-gib 108
```

Two measured size models back this: the 397B harvest form
(`new = base − 1.87 GiB × shallow_bits_harvested`, 6-for-7 within ±0.4 GiB)
and the dense composition form (`total = codes + scales + carry`, closed to
≤0.003 GiB across three builds and two geometries). Harvesting shallow-layer
K back is ~2x the byte-efficiency of stepping down a flat rung — it buys the
sizes between rungs. It is never free and never beats a flat rung at the flat
rung's own size; the pricer tells you both.

## Pipeline

```
fit → verify (outlier gate) → pack → graft (vision) → release checks
    → smoke-generate through the shipped kernel → score
```

```bash
# Install into a DISPOSABLE venv, never a shared/base interpreter: this
# package ships a model runtime, and which copy of a runtime resolves is a
# real source of wrong conclusions (METHODOLOGY.md §5).
python3 -m venv .venv && . .venv/bin/activate
pip install -e .
vqlab selftest        # real pipeline on a tiny synthetic model (<1 min, uses the GPU)

vqlab fit-moe   --family qwen397b --model <bf16-or-base> --out fits/K256 ...
vqlab fit-dense  --src <src> --out fits/d2K512 ...   # dense families
vqlab build-dense --base <q4-base> --mlp fits/d2K512 --out <artifact>
vqlab verify    <artifact> --outlier 3.0        # BEFORE believing any score
vqlab pack      <artifact> --out <packed>       # true bit-width, measured size
vqlab graft     <packed> --vision <bf16>        # MoE vision tower, bf16
vqlab check     <packed>                        # release + bundle gates
vqlab smoke     <packed>                        # one token, fused path, resident
vqlab score     <packed> --instrument referee   # or kl
vqlab manifest  write <packed>                  # stamp provenance
```

Every subcommand is a thin wrapper over a standalone script in
`src/vqlab/`; `vqlab <cmd> --help` shows the full surface. See
[REPRODUCING.md](REPRODUCING.md) for the exact commands behind each paper
table row.

## Requirements

- Apple Silicon Mac; RAM sized to the artifact for fit/smoke (scoring
  streams and can exceed RAM). The 397B fits used 96–128 GB machines.
- Python ≥ 3.12, `mlx`, `mlx-lm` (stock — artifacts bundle their own
  runtime), `numpy`, `safetensors`.

## Verifying your install

**It does real GPU work.** The fits and kernels are genuine, so although it
takes seconds of GPU, it *contends* — do not run it on a machine that is
mid-experiment.

`vqlab selftest` is not a mock: it synthesizes a small checkpoint and runs
the shipped fitter, outlier gate, packer, manifest and Metal kernels over it
as subprocesses, checking what each stage is supposed to guarantee — seeded
fits reproduce, packing is bit-exact, the gate fails a collapsed tensor, the
manifest catches altered bytes, and a dense bundle serves on a stock mlx-lm.
Every gate is exercised in **both** directions, because a gate that only ever
passes tells you nothing. The two stages that need a real multi-GB
checkpoint — end-to-end generation and scoring — are reported as SKIPPED with
the reason, never silently dropped.

## Honesty rules baked into the tool

- The outlier gate refuses to be skipped quietly: scoring an ungated
  artifact is on you, and the docs say what it cost us.
- Sizes are computed from packed shards on disk, never from index metadata.
- The fitter aborts on per-geometry relative-error thresholds and never
  writes in place (fits resume).
- `bundle-accept` tests the runtime copy *lifted from the artifact*, not
  whatever your import path resolves to.

## Known scope limits

- MLX/Metal only; kernel conclusions (threadgroup ceiling, d8 decode tax)
  are Apple Silicon specific.
- Families onboarded: Qwen3.5-397B-A17B, Qwen3.6-35B-A3B (MoE),
  dense Qwen 27B. Gemma-family fitting code is included, but no quality
  claims are made for it: its raw perplexity is invalid as a property of
  the model, which makes scoring non-deterministic.
- New family? Read `docs/ONBOARDING.md` (the two-hour characterisation pass
  to run before fitting anything).

## License

Apache-2.0. If you use VQLab or its artifacts in published work, cite the
companion paper.
