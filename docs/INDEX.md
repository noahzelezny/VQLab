# INDEX — read this before any other doc in vqlab/docs/

37 docs, ~4,600 lines, written across an arc in which several headline numbers
were later found WRONG. A doc is not automatically true because it is committed.
This index says what each doc is FOR and whether it still holds.

**Order of authority when two docs disagree:**
1. `FINDINGS-LOG.md` — the measured record, corrections applied in place.
2. The dated doc for that specific experiment, *if* this index marks it CURRENT.
3. Everything else is history.

---

## START HERE
| doc | status | what it is |
|---|---|---|
| **FINDINGS-LOG.md** | **AUTHORITATIVE** | Every measured result, F-numbered, newest first. Corrections edit entries in place and say CORRECTED. If a number is not here, treat it as unverified. |
| **MORNING-REPORT-2026-09-09.md** | CURRENT | Latest state: what verified, what dissolved, what is open. |
| **KERNEL-COVERAGE.md** | CURRENT | Which geometry each artifact uses and which kernel serves it. Start here for "what shape is this model". |
| **TWO-RUNTIMES.md** | CURRENT, CRITICAL | The lab runtime (`vqlab/src/vqlab/vq_switch.py`) vs the artifact's BUNDLED `model.py` are different files. Measuring one while changing the other has produced two false results (see F31). Read before any benchmark. |

## Experiment docs
| doc | status | note |
|---|---|---|
| CBDEV-ARM-2026-09-08.md | CURRENT, one row corrected | The 1.76x / 1.43x CB_DEV win is **independently verified at 1.46x** (F32). Its `d4-K512 = 0.997x tie` row is WRONG — controlled re-run says **0.83x regression** (F26). |
| D8-BENCH-2026-09-07.md | **PARTLY WRONG** | The d8 promotion numbers stand. Its "NSUB=80 relaxation is a NULL" section is **wrong — it is 1.34x** (F31); that section was measured against a bundle rewritten 7 minutes after the doc. A correction banner sits above it. Every inference drawn from that null is void. |
| RTILE-2026-09-08.md | **CORRECTED** | Its "1.23x single-box / 0.82x ring, the sign flips on topology" framing did not survive. Matched-harness re-runs say RTILE=64 is **uniformly slower** (0.75x on d4-K2048, 0.93-0.97x on Flash-2.1). See F25-corrected. |
| DECODE-DIAGNOSTIC-2026-09-08.md | CURRENT | 99.83% of decode step time is inside mlx-lm's forward. Kills the per-token-collective hypothesis. |
| DECODE-SWARM-2026-09-08.md | CURRENT (method) | Four workers converged on one measurement; that measurement then falsified their shared hypothesis. |
| AFFINE-BASELINE-397B-2026-09-08.md | CURRENT | VQ at 87% of affine on the flagship — the parity target. |
| MOE-PREFILL-ATTRIB / PREFILL-OPT-BENCH / VQGEMM-BENCH / WDEC-BENCH | CURRENT | Earlier prefill work; not contradicted, not re-verified. |
| MTP.md, MTP-VALIDATION, MTP-EXO-SPEC | CURRENT | Speculative decoding. Note MTP costs ~3.4x on multi-request workloads. |
| MTP-USAGE.md | CURRENT | Setup, serving, and the measured speedups; moved out of README.md 2026-09-15. |
| GEMMA-DIVERGENCE, DENSE-VQ-DECODE | CURRENT | Dense path; gemmseg is MoE-only, so MoE prefill numbers do NOT transfer. |

## Method / process
| doc | status | note |
|---|---|---|
| SWARM-ANCHOR-AUDIT-2026-09-08.md | CURRENT, READ IF PROPOSING | 15 of 16 proposals died to reading the source; anchors were accurate while conclusions were not. The failure modes are enumerated. |
| SWARM-PREFILL-2026-09-08.md | CURRENT | Earlier swarm round. |
| ONBOARDING.md, MEMORY-PLAYBOOK.md, PUSH-RUNBOOK.md, CORPORA.md | CURRENT | Operational. |

## History — snapshots, superseded by FINDINGS-LOG
`STATE-2026-09-01/02/03-EOD/07/07-EVENING/08.md`, `MORNING-REPORT-2026-09-03/04.md`
Useful for "what did we believe on date X". Do not cite as current.

## Drafts — unpublished, may contain stale numbers
`CARD-UPDATE-DRAFT.md`, `GLM-27-CARD-DRAFT.md`, `LINKEDIN-DRAFT.md`,
`COMMUNITY-NOTE-27B.md`, `REVIEW_BRIEF.md`, `card-section-prefill-memory.md`

---

## The three errors that cost the most, so they are not repeated
1. **Cross-harness comparison.** Comparing an exo number to a local-probe
   number and attributing the difference to hardware/geometry. Cost: the
   phantom "RTILE geometry flip", plus three wrong causes for a decode
   collapse. **Re-run BOTH arms in ONE harness before concluding.**
2. **Measuring an unapplied change.** A benchmark run against a runtime that
   did not yet contain the change, reported as a confident null. Cost: the
   ragged-NSUB "null" that was really 1.34x. **Verify the change is in the
   file you are loading** (see TWO-RUNTIMES.md).
3. **Trusting an excerpt.** Proposals citing a correct line number and
   inventing its contents. **Read the file.**
