# CLAUDE.md — vqlab

**This repo is a TOOLKIT, not a pile of experiments.** Nearly every question
you are about to answer already has an instrument here. Building a one-off
script in a scratch directory when a CLI command exists is the single most
common failure mode in this repo's history, and it costs hours and produces
results that nobody can reproduce.

## Do these three things before writing any code

```bash
python -m vqlab.cli --help          # 41 commands. Read the list. Twice.
sed -n 1,60p docs/INDEX.md          # what every doc is FOR + whether it still holds
sed -n 1,40p docs/ONBOARDING.md     # the mechanical pass before fitting ANY new family
```

Then grep for the thing you were about to build:

```bash
grep -rli "<the concept>" src/vqlab/ docs/ research/quantlab/
```

## The instruments that get rebuilt by accident

| question | use this | NOT |
|---|---|---|
| which layers deserve more bits? | `vqlab layer-leverage` (per-layer damage vs a bf16 teacher; qwen4_exp supported) | hand-rolled band ablations |
| how much damage does this artifact carry? | `vqlab score` / `kl_damage.py` | ad-hoc KL scripts |
| ppl on the house corpora | `scripts/score_ppl_resident.py` + the THREE corpora in `src/vqlab/referee/` (prose / code-public / literary) | your own corpus files |
| task benchmarks | `research/quantlab/score_tasks_streaming.py` (layer-streamed; scores models larger than RAM) | a new eval harness |
| is this artifact releasable? | `vqlab check-release` / `check-bundle` / `selftest` | eyeballing |
| fit a mixed-geometry rung | `vqlab fit-moe --vq-layers` (scatter fits) | bespoke build scripts |

If an instrument genuinely does not exist, **add it to `src/vqlab/` with a CLI
entry** rather than leaving a script in scratch. That is why the toolkit is
good: every past agent who needed something added it here.

## Authority order when sources disagree

1. **The shipped artifact's own `config.json` and `README.md`** — the record
   of what actually shipped. A card's methodology section says how the mix
   was chosen; the config says what the mix IS.
2. `docs/FINDINGS-LOG.md` — the measured record (F-numbers, corrections
   applied in place).
3. `research/quantlab/EXPERIMENTS.md` — a LAB NOTEBOOK. It narrates attempts,
   including ones whose verdicts the same arc later overturned. **Never
   characterize a released artifact from an experiment entry** (this error
   was made twice in one hour on 2026-09-13; both times a 30-second config
   read would have prevented it).

## Standing rules that have each been paid for

* **Perplexity is deterministic.** Re-scoring an artifact returns the same
  number; it cannot estimate a noise floor. A fit-to-fit floor requires a
  SECOND INDEPENDENT FIT of the same recipe.
* **Rank allocation by KL, not ppl.** Ppl aggregates and absorbs offsetting
  errors. Ppl is the RELEASE GATE; KL is the ranking instrument.
* **One harness.** Never compare a number from one scoring path against
  another. qwen4_exp loglikelihoods even shift 0.1-0.7 nats with batch
  composition (F87) — same path AND same batching.
* **Depth/geometry laws are family-local.** GLM, 397B and Flash each measured
  a different shape. Never inherit an allocation across families.
* **Generate one token through the shipping runtime** before calling anything
  releasable (rule III.11 — an unservable artifact once scored perfectly).
* **Artifacts NEVER go on the internal disk.** Scratch:
  `/Volumes/Thunderbay SSD/vqlab-scratch/`; fits:
  `/Volumes/Thunderbay HDD/vqlab-fits/`; teachers:
  `/Volumes/Thunderbay HDD/Teacher Models/` (archived, not re-downloaded).
* **Version vocabulary:** v1 = first codebooks; v2 = mixed codebooks
  (measured per-layer allocation); v3 reserved for gradient-tuned. Artifacts
  never carry campaign letters.

## Long runs

Overnight/multi-hour work needs `nohup ... & disown` plus per-module
checkpoints — a chain tied to the session dies with it. Geometry refits crash
on GPU timeouts under disk contention; wrap them in a retry supervisor that
resumes from checkpoints.
