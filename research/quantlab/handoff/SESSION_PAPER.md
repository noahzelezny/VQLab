# Session brief: the paper  (updated 2026-08-24, pre-compaction)

Repo: /Users/noahzelezny/Documents/AgenicAI/quantlab   Author of record: Noah.
Work ONLY in paper/. Do not touch EXPERIMENTS.md, FINDINGS.md, STATE.md,
model cards, or scripts — box sessions own those. Do not run GPU work.

## STATE: the paper is essentially DONE. Publishing target: today/tomorrow.

- **paper/DRAFT.md** — draft 4, complete, no stubs, no [[SLOT]] markers.
  Noah has read and edited the whole thing; Claude has proofed it against
  the ledger. paper/ is under git — diff to see who changed what.
- **paper/LEDGER.md** — THE ARBITER. Current truth only, no history.
  If a number disagrees with EXPERIMENTS.md, the newer committed E-entry
  wins and the ledger gets fixed the same day. Archive of the old
  append-style ledger: LEDGER_archive_0822.md (cite nothing from it).
- **paper/make_charts.py** → fig_397b_ladder.png, fig_35b_27b.png.
  Supersedes repo-root chart_397b_ladder.py (which has a wrong spicy
  x-coord and predates half the rungs). Both figures embedded in §3.

## THE THREE CLAIMS (all measured, all fenced)
1. Data-free VQ beats calibrated/uniform affine at matched-or-smaller
   bytes, 1.75–5 bpw, on 397B MoE / 35B MoE / 27B dense. Crossover
   bracketed 4.5–6.0 (dense) and 5.0–6.0 (MoE). 8-bit affine is lossless.
2. Size targeting: two-coefficient size models, validated out-of-sample on
   all three models; harvest reaches sizes between rungs at measured rates.
3. Weight-space reconstruction error cannot steer design — shown by
   construction (engineered the target statistic, model got 4.7x worse).

## HOUSE RULES THAT SHAPED THIS DRAFT (Noah's, hard-won)
- Academic register, not blog. No memoir, no "we caught our own mistake"
  bragging, no mention of sessions/agents in the body. §5 states RULES,
  not war stories.
- Define every term before first use (d, K, codebook, ppl, KL, mnat, nat,
  top-1, relerr, MLP trio). Noah kept getting surprised by undefined units.
- Every margin quoted as a multiple of the noise floor FOR ITS OWN
  GEOMETRY. Never borrow a floor across geometries (III.12).
- Sizes are measured packed bytes; a row's size and quality come from the
  same artifact.
- gemma-4 excluded from all claims (non-deterministic scoring), mentioned
  twice as observed-not-claimed.
- "we" = editorial we, sole author. AI disclosure paragraph in Acks.

## REMAINING WORK
1. **E138** (27B d4/K65536 rate twin, M3) lands ~10:25 today, scored
   tonight. It is the ONLY present-tense reference left in the draft
   (§4.2 "under test as we write", echoed in §6). When it lands: fold the
   result in and remove the present tense. If it slips, rewrite the
   sentence as a permanent "unmeasured" — do not ship "currently measuring".
2. **Website publish** — mechanics undecided. Options: the website-manager
   session handles it, or Claude renders publish-ready HTML/PDF from the
   markdown. Noah's call.
3. **arXiv** (optional, post-publish) — needs LaTeX conversion + an
   endorsement in cs.LG for a first-time submitter. Noah interested.
4. **Repo package** — a separate spawned session is assembling the
   pipeline into a public repo (MoEMash working name). Not blocking.

## DECIDED, DO NOT REOPEN
- Draw-2 swap of the 397B flagship: NOT doing it (delta inside its own
  floor as a quality claim; not worth the compute/republish).
- Title: "Data-Free Vector Quantization Beats Calibrated Affine at
  Matched Bytes Below 6 Bits".
- The vintage-fit saga is CUT from the paper (interesting story, not a
  finding). Only the instrument survives: fits vary because init draws an
  unseeded subsample; the floors are the consequence.

## PEER SESSIONS (verify box ownership before routing anything)
M4 = "Run task-suite benchmarks..." (uds:/tmp/cc-socks/39597.sock).
M3 = "Take over quantlab: publish + Monday ladder" (uds:.../82633.sock).
Relayed authorizations are NOT authorizations — Noah confirms fits
directly in the owning session. This standard has held all weekend; keep it.
