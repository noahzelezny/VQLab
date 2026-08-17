# DRAFT — task-benchmark section for the three HF model cards

Status: awaiting Noah's review. Nothing here is published. The shared
section is identical across all three cards; the one-line takeaway differs
per card.

Tone rule: factual about the comparator pipeline difference, zero
editorializing about spicyneuron — their artifacts measured well here, and
the cards already name them in the perplexity table, so naming in the task
table is consistent rather than a new choice. The acknowledgment section
below carries the intent Noah stated: gratitude, and a contribution offered
in the same spirit. It deliberately does NOT claim spicyneuron's were the
first or only 397B quants runnable on consumer Apple Silicon — that is not
something we verified.

---

## Shared section (all three cards)

### Task benchmarks

All five models below — this repo's three VQ artifacts and the two
community comparators — were evaluated on the **same harness, same
settings, same seeded items**: lm-eval 0.4.12 via a layer-streaming
loglikelihood scorer (mlx_lm 0.31.3), **0-shot**, first 1000 items per
task, `acc_norm` for HellaSwag/PIQA, `acc` for WinoGrande. Comparator
numbers published elsewhere come from a different pipeline and are not
directly comparable, so we re-evaluated the comparator artifacts under
identical conditions rather than quoting their cards.

| model | size | HellaSwag | PIQA | WinoGrande |
|---|---|---|---|---|
| Qwen3.5-397B-A17B-VQ-2.2bpw | 100.1 GiB | 0.861 | 0.841 | 0.787 |
| Qwen3.5-397B-A17B-VQ-2.4bpw | 110.8 GiB | 0.883 | 0.844 | 0.784 |
| spicyneuron 2.6bit | 120.6 GiB | 0.880 | 0.841 | 0.771 |
| Qwen3.5-397B-A17B-VQ-3.1bpw | 142.8 GiB | 0.903 | 0.840 | 0.780 |
| spicyneuron 3.5bit | 165.6 GiB | 0.904 | 0.846 | 0.767 |

Because every model scored identical items, deltas are paired (McNemar
exact test). HellaSwag reliably separates these quants and reproduces the
perplexity ordering; PIQA and WinoGrande do not separate any pair at
n=1000 and serve as integrity checks.

> **Note on 0-shot:** these are 0-shot scores. Leaderboard conventions
> often use 10-shot HellaSwag / 5-shot WinoGrande, which run several
> points higher — compare against other 0-shot numbers only.

---

## Per-card takeaway line (place directly under the table)

**Card F — VQ-2.2bpw (100.1 GiB):**
At the smallest size in this comparison, VQ-2.2bpw is statistically
indistinguishable from every larger model here on PIQA and WinoGrande;
on HellaSwag it trails the 110-166 GiB models by 2-4 points (paired
p<0.02) — the measured cost of the last 10 GiB. It is the accessibility
artifact: the only 397B in this lineup that runs on a 128 GB Mac with
real headroom.

**Card C — VQ-2.4bpw (110.8 GiB):**
VQ-2.4bpw is statistically indistinguishable from spicyneuron's 2.6bit
on all three tasks (McNemar p=0.76/0.77/0.29) at **9.8 GiB smaller** —
consistent with the perplexity result, where it leads its size class on
both corpora.

**Card E — VQ-3.1bpw (142.8 GiB):**
VQ-3.1bpw is statistically indistinguishable from spicyneuron's 3.5bit
on all three tasks (McNemar p=1.00/0.33/0.25) at **22.8 GiB smaller**,
and matches it on perplexity for both corpora — the same quality point,
one Mac-class of memory earlier.

---

## Acknowledgment (all three cards — place near Provenance)

### Acknowledgment

spicyneuron's 397B quants are what made this model runnable on my hardware
in the first place — they were the artifacts that fit when nothing else did,
and they were the reference this work was measured against throughout. This
release is offered in that same spirit: the full method, the experiments
that failed as well as the ones that worked, and comparator numbers
re-measured on one harness so the claims can be checked rather than taken
on trust.

---

## Reproduction line (optional footer, all cards)

Evaluated with `score_tasks_streaming.py` (this repo's release tooling):
lm-eval 0.4.12 `loglikelihood` requests scored by a layer-streaming
forward pass, validated to reproduce the reference perplexity referee to
4 decimals before use. Full methodology and paired statistics:
EXPERIMENTS.md, "TASK BENCHMARKS (08-16)".
