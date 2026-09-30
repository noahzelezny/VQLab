# docs/ — what each document is for

Several headline numbers in this project were later found wrong. A document is
not true because it is committed; this index says what each one is FOR.

**When two sources disagree:** a shipped artifact's own `config.json` wins,
then `FINDINGS-LOG.md` (the measured record, corrections applied in place),
then everything else.

## Start here

| doc | what it is |
|---|---|
| [FINDINGS.md](FINDINGS.md) | The law book: settled laws, retracted leads (do not re-chase), instrument rules, MLX/Metal rules, open questions. Read before proposing any experiment. |
| [FINDINGS-LOG.md](FINDINGS-LOG.md) | Every measured result, F-numbered, newest first. If a number is not here, treat it as unverified. |
| [ONBOARDING.md](ONBOARDING.md) | The mechanical pass before fitting a new model family (`vqlab onboard`). |
| [CORPORA.md](CORPORA.md) | The three referee corpora: provenance, licensing, why they are frozen. |

## Methods

| doc | what it is |
|---|---|
| [ALLOCATION-METHOD.md](ALLOCATION-METHOD.md) | Direct codebook sensitivity allocation: how per-layer geometry is chosen. |
| [GEOMETRY-CAMPAIGN.md](GEOMETRY-CAMPAIGN.md) | The Flash geometry campaign (F67–F88) as one readable account. |
| [KL-TUNED-CODEBOOKS-PLAN.md](KL-TUNED-CODEBOOKS-PLAN.md) | KL-tuned code re-selection and the additive-codebook work, with verdicts. |
| [PROVENANCE.md](PROVENANCE.md) | Build records (`vqlab_provenance.json`): what every artifact records about how it was made. |

## Runtime

| doc | what it is |
|---|---|
| [TWO-RUNTIMES.md](TWO-RUNTIMES.md) | The repo runtime vs an artifact's bundled `model.py`: which copy actually runs. Read before any benchmark. |
| [RUNTIME-SETTINGS.md](RUNTIME-SETTINGS.md) | Every runtime setting that makes these artifacts runnable, in one place. |
| [RUNTIME-SHIP-PLAN.md](RUNTIME-SHIP-PLAN.md) | Which runtime profile (v1.5 / v2) ships with which weights. |
| [V2-RUNTIME.md](V2-RUNTIME.md) | The v2 runtime: what it is and how it ships. |
| [KERNEL-COVERAGE.md](KERNEL-COVERAGE.md) | Which Metal kernel serves each module geometry. |
| [SKIPZERO.md](SKIPZERO.md) | Dead-row skipping: the 397B's all-zero rows stored and served compactly. |
| [MEMORY-PLAYBOOK.md](MEMORY-PLAYBOOK.md) | Peak memory too high? Measured fixes, in order. |
| [DENSE-VQ-DECODE.md](DENSE-VQ-DECODE.md) | Root causes of the dense 27B decode problems. |
| [GEMMA-DIVERGENCE-2026-09-07.md](GEMMA-DIVERGENCE-2026-09-07.md) | Forensics of a two-path perplexity divergence on gemma-26b. |

## Speculative decoding

| doc | what it is |
|---|---|
| [MTP-USAGE.md](MTP-USAGE.md) | Using the multi-token-prediction heads, with measurements. |
| [MTP.md](MTP.md) | Per-family MTP findings and open questions. |

## Models

| doc | what it is |
|---|---|
| [model-cards/](model-cards/README.md) | The published model cards and the rules they follow. |
| [MODEL-FAMILY-LEDGER.md](MODEL-FAMILY-LEDGER.md) | Released rungs per family, with config-verified geometry and every score on record. |
| [CANONICAL-397B-NUMBERS.md](CANONICAL-397B-NUMBERS.md) | The single source for 397B figures quoted on cards and in the paper. |
