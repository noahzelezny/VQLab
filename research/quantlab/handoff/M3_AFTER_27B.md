# M3 queue — after the 27B candidate finishes  (written 2026-08-24 by the paper session)

Nothing here is authorized to start until the 27B run in flight is done.
Nothing here blocks tonight's paper post.

## 1. [DEFERRED BY NOAH — 08-22 and again 08-24. Do not re-propose.]
##    Re-run the 397B task suite for TWO artifacts  (~3 h)

**Noah has declined this twice.** The live cards' stale-row labels stand as
the correct interim state; nothing is blocked and the paper does not depend
on it. Kept below only so the work is specified if he ever asks for it.

The Aug-16 sweep (`results_tasks/`) predates two of the three artifacts it
scored. Live cards now LABEL the stale rows; this replaces them.

| artifact | benched Aug 16 | published now | action |
|---|---|---|---|
| VQ-2.2bpw | v1 flat d4/K128 | d8/K16384 (since 08-22) | **re-run** |
| VQ-2.4bpw | flat K256 | unchanged | leave — 08-20 runtime change is inert on the prefill-shaped scoring path |
| VQ-3bpw   | old 3.1 fit    | E91 flat d4/K2048 (published 08-24) | **re-run** |

- Harness unchanged: `./run_task_bench.sh` (lm-eval 0.4.12 via
  `score_tasks_streaming.py`, 0-shot, limit 1000, batch 256). It is RESUMABLE
  and SKIPS any model whose results JSON already exists — so **move or delete
  the two stale JSONs first**, or the run is a no-op:
      results_tasks/Qwen3.5-397B-A17B-VQ-2.2bpw.json (+ .samples.json)
      results_tasks/Qwen3.5-397B-A17B-VQ-3.1bpw.json (+ .samples.json)
  Archive them, do not delete — they are the provenance for the labelled rows.
- MODELS_ROOT paths in the script are the OLD staging names and no longer
  exist. Point them at the real build dirs:
      2.2bpw -> rotlab--397B-d8K16384-packed
      3bpw   -> rotlab--397B-flatk2048-refit-packed
  and name the outputs for the CURRENT repos (…-VQ-2.2bpw, …-VQ-3bpw).
- Expected cost, from the Aug-16 log: 3bpw ~4600 s; 2.2bpw d8 ~6100 s
  (K128 was 5135 s; E115 measured d8 ~19% slower decode). ~3 h total.
- RUN ON M3. The workload is compute-bound on VQ decode, not IO: the LARGEST
  artifact was the FASTEST (143.7 GiB in 4600 s vs 100.9 GiB in 5135 s), and
  an affine model of similar size finished in 610 s. M4 would also have to
  stream every byte over SMB.
- **Record what you scored.** `results_tasks/*.json` stores only a directory
  NAME — no hash, no size. That is exactly why this staleness was invisible;
  it is the same name/bytes divergence as E94 and the Aug-19 base rewrite.
  Do NOT invent a second stamping scheme: `artifact_manifest.py` already
  exists for this and was written after those two incidents. Stamp each
  artifact (`./artifact_manifest.py write <dir>`) and have the results writer
  record the manifest id, which closes this for every results file at once.
  (Suggested by the public-repo session; adopted. Caveat worth knowing: the
  manifest hashes the FIRST 1 MiB per shard plus bytes+mtime — an identity
  stamp, not a full-content hash. For the publish check, keep using full
  sha256 against the remote LFS oid.)

Then: update the task tables + remove the stale-row note in
MODEL_CARD_397B_C.md, _F.md, and add the table to _G.md (which currently
says "not yet re-measured"), and push with `push_card_fixes.py`.
NOTE: that script's map deliberately has NO 3.1bpw entry — HF redirects the
renamed id, and pushing by it overwrote card G once already. Add a
`Qwen3.5-397B-A17B-VQ-3bpw -> MODEL_CARD_397B_G.md` entry instead.

## 2. 35B release prep — TWO artifacts (Noah deciding; do not publish unasked)

Target set (recommendation from the paper session, pending Noah).
**Cited by artifact directory, NOT by E-number** — E140 and E141 each name two
different experiments (see the collision note at the end of this file):
  - **`e94b-35b-K8192-refit-0821-packed`**  d4/K8192, 14.838 GiB text-only -> ~15.67 grafted
  - **`e140-35b-d2K1024-packed`**  d2/K1024, 21.394 GiB text-only -> ~22.22 grafted
Explicitly NOT shipping: `e141-35b-d2K4096-packed` (d2/K4096, sits ABOVE the affine frontier —
1.91x worse than q6 at 1.1 GiB smaller; we would be publishing a rung the
paper reports as a loss), flat d2/K256 (KL and ppl disagree about whether it
beats the published 18.7, and its 36.862 predates E141 in the record while
its 17.643 size comes from E141 — same split-provenance shape e94b had),
and R2 d4/K16384 (does not dominate e94b: 0.9 GiB larger, better KL but
worse ppl).

Per artifact, before anything is published:
  1. graft the vision tower (both published 35B builds include it; every lab
     rung on disk is text-only, 0 vision tensors)
  2. re-gate the GRAFTED bytes (outlier gate) — grafting changes the artifact
  3. KL re-score the grafted artifact and confirm the text path is identical
     to the text-only build (the gemma card's precedent)
  4. III.11 smoke in a STOCK venv, on the copy that ships
  5. `check_release.py`, then a card
Recompute the bpw label from the grafted size — the published naming is by
bpw (13.8 GiB = "3.4bpw", 18.7 = "4.6bpw"), not by geometry.

## 3. Standing rules that bit us today

- A renamed HF repo REDIRECTS. Remove it from any push map; do not rely on
  the redirect. (Card G was overwritten this way and restored.)
- Verify publishes by sha256 against the remote LFS oid and by fetching the
  live card — never by the uploader's progress output, which reported healthy
  progress for 74 minutes while committing nothing (logs_live_upload_101).
- Residue files (`*.pre_*`, `__pycache__`) must be EXCLUDED via
  ignore_patterns, never moved mid-upload.
- **Read back after write.** A push tool that reports success from its own
  exit code structurally cannot see an overwrite that happened through a
  redirect — the only check that catches it is fetching the live card
  afterward. Same reason the outlier gate runs against the bytes on disk
  rather than the fitter's log. (Framing from the public-repo session.)

## 4. TRAP: do not measure a floor with the dense fitter at defaults

`fit_dense_vq.py` has seeded by default (`--seed 1234`) since E139 (08-22
22:19). A naive "fit it three times and quote the range" loop now returns a
floor that is NOT the draw distribution — and it does not return zero either,
which would at least look broken. E139 measured seeded-vs-seeded divergence at
**0.0100% of codes** (a second nondeterminism source survives the RNG pin;
Metal reduction order is the untested candidate). So the loop yields a small,
plausible, nonzero number that would certify every third-decimal margin as
real. It fails in the flattering direction.

**To measure a floor, pass `--seed -1`** (E139 says so explicitly) or use
distinct seeds per draw.

Floors already in the paper are SAFE: the dense 2.085/0.0447 came from E127,
whose three draws were built 08-22 00:46-01:56, ~20 h before the seed landed
(verified against the commit timestamp). The 35B and 397B fitters
(`vq_35b_codes.py`, `vq_397b_codes.py`) have no seed at all and are
deliberately untouched.

**Forward caveat for the paper:** E138 was relaunched SEEDED. The draft now
states that every artifact in it is a single unseeded draw. If E138 is folded
into §4.2, that sentence needs qualifying — a seeded fit is not a draw from
the same distribution the floors describe, and its margin against an unseeded
floor is CONSERVATIVE, not flattering. Same applies to any same-seed A/B
(e.g. the E142 iters arms): a shared seed removes between-arm draw variance,
so the unseeded floor is too wide a bar for it, not too narrow.

## 5. E-NUMBER COLLISION — arbiter ruling (paper session, 08-24)

E140 and E141 were each minted twice by two sessions allocating concurrently.
Verified in the current file:
  - `## E140` (8656) is the M3's box-clustering RETRACTION
  - the M4's E140 (35B d2/K1024) has **NO heading in EXPERIMENTS.md at all** —
    it lives in paper/LEDGER.md and as artifact dirs
  - `## E141` appears THREE times (8709 M4 pre-reg; 8824 + 8868 M3 init-starvation)
  - lines 8713/8722/8729/8737/8754/8762 cite "E140" meaning the M4's rung

So one reference resolves to the WRONG experiment and the other resolves to
NOTHING. That is III.7 and III.10 in the same pair of numbers.

**RULING — suffix, do not renumber.** `E140-M3` / `E140-M4`, `E141-M3` /
`E141-M4`. Renumbering the M4's breaks artifact directory names on disk, which
are load-bearing and whose mid-flight mutation is the 101 GiB lesson;
renumbering the M3's breaks commit messages, which are immutable. Suffixing
costs heading edits.

**One addition to the M3's proposal:** suffixing headings is not sufficient,
because E140-M4 has no heading to suffix. It needs a real stub entry pointing
at where its result lives, or `E140-M4` becomes a fresh III.10 phantom — a
citation that looks valid and resolves to nothing.

**APPLIED BY THE M3, NOT BY ME.** Two sessions writing one file without
coordination is the root cause; a third session editing it now repeats the
pattern. I rule, the file's owner applies.

**Allocation going forward: per-session RANGES, not a central allocator.**
The M3 offered me the allocator role; I decline it. A single allocator is a
bottleneck and a single point of failure — this session can be compacted or
end mid-day, and then nobody can mint a number. Ranges are lock-free and
survive any session dying. Suggested: M3 140-179, M4 180-219, paper 220-239,
recorded in the EXPERIMENTS.md header so the rule is where the numbers are.
