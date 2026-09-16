# VQLab

**Size-targeted vector-quantized (VQ) builds of large models on Apple Silicon,
with MLX.** Name a byte budget, price the recipe before fitting, fit it
data-free, pack to true bit-width, verify, and serve through stock `mlx-lm`.
The runtime ships inside the artifact.

Paper: *Data-Free Vector Quantization Beats Affine Quantization at Matched
Bytes Below 6 Bits*, doi:[10.5281/zenodo.22119017](https://doi.org/10.5281/zenodo.22119017).
Artifacts: [TheDrainFlorist](https://huggingface.co/TheDrainFlorist) on
Hugging Face.

## Published models

Every artifact this tool's pipeline has shipped (20 repos, live list on the Hub):

| model | family |
|---|---|
| [Qwen3.5-397B-A17B-VQ-2.2bpw](https://huggingface.co/TheDrainFlorist/Qwen3.5-397B-A17B-VQ-2.2bpw) | 397B MoE (d8/K16384) |
| [Qwen3.5-397B-A17B-VQ-2.4bpw](https://huggingface.co/TheDrainFlorist/Qwen3.5-397B-A17B-VQ-2.4bpw) | 397B MoE |
| [Qwen3.5-397B-A17B-VQ-2.6bpw](https://huggingface.co/TheDrainFlorist/Qwen3.5-397B-A17B-VQ-2.6bpw) | 397B MoE |
| [Qwen3.5-397B-A17B-VQ-3.1bpw](https://huggingface.co/TheDrainFlorist/Qwen3.5-397B-A17B-VQ-3.1bpw) | 397B MoE (flagship) |
| [Qwen3.6-35B-A3B-VQ-3.4bpw](https://huggingface.co/TheDrainFlorist/Qwen3.6-35B-A3B-VQ-3.4bpw) | 35B MoE |
| [Qwen3.6-35B-A3B-VQ-3.8bpw](https://huggingface.co/TheDrainFlorist/Qwen3.6-35B-A3B-VQ-3.8bpw) | 35B MoE |
| [Qwen3.6-35B-A3B-VQ-4.6bpw](https://huggingface.co/TheDrainFlorist/Qwen3.6-35B-A3B-VQ-4.6bpw) | 35B MoE |
| [Qwen3.6-35B-A3B-VQ-5.4bpw](https://huggingface.co/TheDrainFlorist/Qwen3.6-35B-A3B-VQ-5.4bpw) | 35B MoE |
| [Qwen3.8-27B-VQ-3.9bpw](https://huggingface.co/TheDrainFlorist/Qwen3.8-27B-VQ-3.9bpw) | dense 27B |
| [Qwen3.8-27B-VQ-4.5bpw](https://huggingface.co/TheDrainFlorist/Qwen3.8-27B-VQ-4.5bpw) | dense 27B |
| [Qwen3.8-27B-VQ-4.8bpw](https://huggingface.co/TheDrainFlorist/Qwen3.8-27B-VQ-4.8bpw) | dense 27B |
| [Qwen3.8-Flash-Next-VQ-2.1bpw](https://huggingface.co/TheDrainFlorist/Qwen3.8-Flash-Next-VQ-2.1bpw) | Flash-Next MoE (v2, per-layer) |
| [Qwen3.8-Flash-Next-VQ-3.2bpw](https://huggingface.co/TheDrainFlorist/Qwen3.8-Flash-Next-VQ-3.2bpw) | Flash-Next MoE (v2, per-layer) |
| [Qwen3.8-Flash-Next-VQ-4.4bpw](https://huggingface.co/TheDrainFlorist/Qwen3.8-Flash-Next-VQ-4.4bpw) | Flash-Next MoE |
| [Qwen3.8-Flash-Next-VQ-5.5bpw](https://huggingface.co/TheDrainFlorist/Qwen3.8-Flash-Next-VQ-5.5bpw) | Flash-Next MoE |
| [GLM-5.3-Flash-VQ-2.7bpw](https://huggingface.co/TheDrainFlorist/GLM-5.3-Flash-VQ-2.7bpw) | GLM-5.3-Flash MoE |
| [GLM-5.3-Flash-VQ-3.1bpw](https://huggingface.co/TheDrainFlorist/GLM-5.3-Flash-VQ-3.1bpw) | GLM-5.3-Flash MoE |
| [GLM-5.3-Flash-VQ-3.6bpw](https://huggingface.co/TheDrainFlorist/GLM-5.3-Flash-VQ-3.6bpw) | GLM-5.3-Flash MoE |
| [gemma-4-26b-a4b-it-VQ-6.2bpw](https://huggingface.co/TheDrainFlorist/gemma-4-26b-a4b-it-VQ-6.2bpw) | gemma MoE * |
| [gemma-4-e4b-it-VQ-PLE](https://huggingface.co/TheDrainFlorist/gemma-4-e4b-it-VQ-PLE) | gemma dense * |

\* Released and usable, but no quality claims are made for the gemma family
anywhere in this repo or the paper: its scoring instrument is
non-deterministic (see "Known scope limits"). All sizes are measured packed bytes; every margin is stated against
a measured seed-noise floor; see [METHODOLOGY.md](METHODOLOGY.md) for the
rules that keep these numbers honest.

## Results in one table

Streaming-referee perplexity on the 397B MoE, prose and code corpora.
Sizes are packed bytes on disk.

| build | GiB | prose ppl | code ppl |
|---|---|---|---|
| d8/K16384 | 100.97 | 3.0591 | 2.6728 |
| flat K256 (2.4bpw) | 111.62 | 2.7655 | 2.6383 |
| **flat K512** | **122.31** | **2.5634** | **2.6123** |
| flat K2048 (flagship) | 143.68 | 2.3410 | 2.5963 |
| calibrated 2.6-bit comparator | 120.6 | 3.1843 | 2.6667 |
| calibrated 3.5-bit comparator | 165.6 | 2.3614 | 2.6005 |

- flat K512 beats the calibrated 2.6-bit build by 0.62 prose ppl (24x the
  seed-noise floor) at +1.7 GiB.
- d8/K16384 beats the same comparator while 19.6 GiB smaller. It is the
  ~101 GiB build a 128 GB Mac holds.
- The flagship is 21.9 GiB smaller than the calibrated 3.5-bit build with
  prose better by 3.6x the floor and code a tie.
- 35B MoE d4/K8192: 53.0 mnats KL to bf16 vs the community 4-bit's 78.6,
  a 32% reduction. No size claim for this pair (mismatched vision-tower
  bases).
- Dense 27B d2/K512: beats 4-bit affine by 28% KL at 4-bit-class size. The
  recipe is not an MoE phenomenon.

Where the method ends: wins are measured at 2.0 to 4.5 bpw across three
families, the VQ/affine crossover sits at 4.5 to 6.0 bpw on the dense 27B,
and by 8 bits affine is lossless. Prefill is ~0.5x affine at 35B.

Every margin is quoted against a measured seed-noise floor; the code corpus
is private and the vision-tower size offset is disclosed rather than
restated. The full record, with corrections applied in place, is
[docs/FINDINGS-LOG.md](docs/FINDINGS-LOG.md). The rules are
[METHODOLOGY.md](METHODOLOGY.md).

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

# MoE families (Qwen3.5/3.6-class): fit against an affine skeleton + bf16 source
vqlab fit-moe --family qwen3_5 --base <affine-skeleton> --src <bf16> \
    --vq-layers 0-56 --k 256 --dim 4 --out fits/K256
vqlab pack  --src fits/K256 --out artifacts/K256-packed
vqlab graft --artifact artifacts/K256-packed --src <bf16>   # vision tower

# Dense families: fit the MLP trio, splice into a quantized base
vqlab fit-dense --family qwen3_8_dense --src <bf16> --k 512 --dim 2 \
    --out fits/d2K512
vqlab build-dense --family qwen3_8_dense --base <q4-base> \
    --mlp fits/d2K512 --out assembled --dry-run   # then without --dry-run
vqlab pack-dense --src assembled --out artifacts/d2K512-packed

# Gates — run these before believing anything (see METHODOLOGY.md)
vqlab verify --artifact fits/K256 --src <bf16> --family qwen3_5 \
    --outlier 3.0                       # BEFORE any score is believed
vqlab check artifacts/K256-packed       # release + bundle gates
vqlab smoke artifacts/K256-packed       # generate through the SHIPPED runtime

# Score (referee ppl streams > RAM; KL needs a teacher cache)
vqlab score --model artifacts/K256-packed
vqlab kl cache --model <bf16> --out-dir caches/family
vqlab kl score --model artifacts/K256-packed --cache-dir caches/family

vqlab manifest write artifacts/K256-packed   # stamp provenance
```

`vqlab --help` lists all commands; `vqlab <cmd> --help` shows each surface.
[REPRODUCING.md](REPRODUCING.md) maps every paper table to its commands.

## Per-layer allocation (v2 artifacts)

v2 artifacts carry a measured per-layer geometry instead of one flat rung.
The process is three commands: `layer-leverage` ranks layers by the jump in
trajectory damage, `alloc-sweep` measures the cost and value curves and
prints the marginal-ppl-per-100MB frontier, `geo-build` refits only the
named modules from the bf16 teacher and keeps every other shipped byte.

## MTP speculative decoding

`vqlab serve` and `vqlab mtp-generate` decode with a multi-token-prediction
head as the drafter. Setup, supported families, and the measured speedups
are in [docs/MTP-USAGE.md](docs/MTP-USAGE.md); per-family findings in
[docs/MTP.md](docs/MTP.md).

## Requirements

Apple Silicon Mac with RAM sized to the artifact for fit and smoke (scoring
streams and can exceed RAM; the 397B fits used 96 to 128 GB machines).
Python 3.12 or newer, `mlx`, `mlx-lm`, `numpy`, `safetensors`.

Install into a disposable venv, never a shared interpreter: the package
ships a model runtime, and which copy of a runtime resolves has produced
wrong conclusions before (METHODOLOGY.md §5).

`vqlab selftest` runs the real fitter, outlier gate, packer, manifest and
Metal kernels over a small synthetic checkpoint, exercising every gate in
both directions. It uses the GPU; do not run it on a box mid-experiment.
The two stages that need a real checkpoint, generation and scoring, report
as SKIPPED with the reason.

## Scope limits

- MLX/Metal only. Kernel conclusions are Apple Silicon specific.
- Families onboarded: Qwen3.5-397B-A17B, Qwen3.6-35B-A3B, dense Qwen 27B,
  Qwen3.8-Flash-Next, GLM-5.3-Flash. Gemma fitting code ships but no
  quality claims are made for it; its scoring instrument is
  non-deterministic.
- New family: read [docs/ONBOARDING.md](docs/ONBOARDING.md) first.

## Layout

- `src/vqlab/` — the toolkit. Every command is a standalone script.
- `docs/` — start at [docs/INDEX.md](docs/INDEX.md), which says what each
  doc is for and whether it still holds.
- `research/quantlab/` — the frozen research tree (experiments E60 to E147,
  the ladders, the paper drafts). [FINDINGS.md](research/quantlab/FINDINGS.md)
  there is the law book: settled laws, retracted leads, instrument rules.
- `AGENTS.md` — instructions for coding agents working in this repo.

## Development note

Assembled with AI assistance under the author's direction and review; the
measurements, the artifacts and every published claim are the author's.

## License

Apache-2.0 for code. Corpus data files carry their own terms
(CC BY-SA for the WikiText prose corpus, public domain for the literary
corpus); see [docs/CORPORA.md](docs/CORPORA.md). If you use VQLab or its
artifacts in published work, cite the companion paper.
