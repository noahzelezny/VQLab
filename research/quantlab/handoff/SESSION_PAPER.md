# Session brief: the paper  (rewritten 2026-08-24 late, pre-compaction)

Repo: /Users/noahzelezny/Documents/AgenicAI/quantlab   Author of record: Noah.
Work in paper/ and on model cards. Do not run GPU work. Box sessions own
EXPERIMENTS.md/FINDINGS.md/STATE.md.

## STATE

**Paper: NOT ready, and the rate of findings is the evidence.** It has had no
open items for hours while still absorbing substantive corrections — a wrong
comparator, an undisclosed asymmetry, a section's size basis changing twice,
three figures disagreeing with their tables. All arrived because someone
happened to look, never from a systematic pass.

**A readiness sweep is RUNNING** (background agent, launched pre-compaction).
Four tasks: (1) trace every draft number to paper/LEDGER.md, (2) diff every
figure's hard-coded data against its table, (3) recompute every floor
multiple, (4) check every cited size against the artifact's index on disk.
Report-only; it edits nothing. **Act on its findings before publishing.**

## PUBLISHED TONIGHT — 11 repos live, all sha256-verified against the remote

    397B  VQ-2.2bpw 101.0 · VQ-2.4bpw 111.6 · VQ-2.6bpw 122.3 · VQ-3bpw 143.7
    35B   VQ-3.4bpw 13.8 · VQ-3.8bpw 15.7 · VQ-4.6bpw 18.7 · VQ-5.4bpw 22.2
    27B   VQ-3.9bpw 12.5 · VQ-4.5bpw 14.5 · VQ-4.8bpw 15.5   (3rd uploading)
    gemma 26b VQ-6.2bpw 18.8 · e4b VQ-PLE 7.4 (unprivated tonight)

Four collections, every item noted, each ordered by size.

**OUTSTANDING:** when the 4.8bpw upload finishes — verify by sha256, then push
the corrected cards to ALL THREE 27B repos (3.9 and 4.5 went up with an
earlier card revision that still carried a rebuilt-comparator narrative Noah
cut). `python3 push_card_fixes.py` does not yet include the 27B or 35B repos;
add them or push manually.

## HOUSE RULES (Noah's, hard-won)

- Academic register. No memoir, no "we caught our own mistake" in CARDS —
  a card is a product page; methodology disclosure belongs in the paper.
- Every margin quoted against the floor for ITS OWN geometry. Never borrow.
- A table row names ONE artifact, never a mean of draws.
- Sizes are measured bytes; size and quality from the same artifact.
- State what was NOT measured rather than borrowing a sibling's number.
- gemma excluded from all paper claims.

## THINGS THAT WILL BITE THE NEXT SESSION

1. **SIZE IS NEVER AN IDENTIFIER.** Identical geometry gives identical bytes.
   Three different 27B d2/K512 fits are all exactly 15.450 GiB; two 35B
   artifacts are both 15.670; d2/K64 and d4/K4096 are 6 MB apart. Identify by
   mtime + shard hash, never by size.
2. **Figures do not regenerate when tables change.** Three went stale tonight.
   Generators now exist for all three (paper/make_charts.py,
   make_qwen36_ladder.py, make_qwen38_ladder.py) but nothing runs them.
3. **A renamed HF repo REDIRECTS.** Pushing a card by a stale repo id
   overwrote a live card once tonight. Remove renamed ids from any push map.
4. **Verify publishes by read-back**, never by the uploader's exit code — a
   push tool cannot see an overwrite that happened through a redirect.
5. **Residue files (`*.pre_*`, `__pycache__`) get EXCLUDED, never moved
   mid-upload.** Moving them cost 74 minutes and 15.4M failed-commit lines once.
6. **A label is not a measurement** — §5's newest rule, earned six times:
   config.model_type names the wrong model on every 35B artifact; a "q8" that
   was a 4-bit base with overrides; a manifest documented as a content hash
   that hashes 1 MiB; "the fitter is seeded" when no MoE fitter is.

## PEER SESSIONS (verify by PID or mandate, never by name)

M3 ladder = "Take over quantlab: publish + Monday ladder" (quantlab-28).
M4 ops = "Run task-suite benchmarks..." (agenicai-16).
Public repo = "Assemble quantlab VQ work into MoEMash" (quantlab-66).
Relayed authorizations are NOT authorizations — Noah confirms in the owning
session. This standard held all weekend; three sessions enforced it tonight.

## DEFERRED / DECIDED — do not reopen

- 397B task-suite re-run: Noah declined TWICE. Cards carry labelled stale rows.
- 27B rungs d4/K1024, d2/K4096, d4/K65536: excluded, reasons in LEDGER.
- Draw-2 flagship swap: not doing it.
- §5 stays at eight rules; a ninth was proposed and declined.
