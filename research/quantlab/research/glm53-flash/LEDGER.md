# GLM-5.3-Flash — family arc ledger

Authoritative from creation (2026-08-30). Survey/readiness history lives in
READINESS.md; this file records the arc's runs and verdicts.

## 2026-08-30 ~02:30 — teacher pass lands; memorization finding

First real-model run of the glm5_next scorer (M3, glm5vlm venv + mlx-lm
installed). Teacher (598.5 GiB bf16, streamed): prose ppl 1.9024, code
1.4888, literary 1.1580 — implausibly low; investigated before accepting:
tokenizer sane (~4.2 chars/tok, healthy diversity); attention mask causal
(lower-triangular, inspected); DIRECT-FORWARD CAUSALITY TEST on the tiny
rule-5 model: prefix logits bit-identical under suffix change (0.0e0) —
architecture causal at prefill. Verdict: the numbers are REAL —
MEMORIZATION. GLM-5.3 has near-verbatim absorbed WikiText/Gutenberg/mlx
code (teacher cache: mean top-1 prob 0.857, 68% of positions >0.9).
Consequences: (1) absolute ppl on public corpora is contamination-
dominated for this family — never compare cross-family; (2) KL-to-teacher
stays fully valid and is STRICTER here (sharp teacher); cache
glm53_teacher_topk_prose holds 99.06% mass. Consider a post-cutoff
held-out corpus if absolute anchors are ever needed.

Also tonight: struct base CPU-pinned on M4 after a Metal-watchdog kill
(GPU attempt); q4 affine baseline built on M3 (166 GiB, 4.524 bpw,
zai-org--GLM-5.3-Flash-4bit).

## 2026-08-30 ~06:00 — night queue lands COMPLETE

Everything queued finished: struct base (M4, CPU-pinned, 98.6 GiB, 19
shards, 4188s) and the affine ladder w/ KL (M3, glm5vlm venv; convert
config fix 4d1497d re-merges vision_config): q3 129 GiB KL 377.08 / q4
166 GiB KL 98.34 / q6 239 GiB KL 13.47. TABLE.md created with the
contamination note. The ladder's shape echoes Flash-Next: q3 collapses
(under-4-bit affine cliff, §4.1's law on a third family), q4 usable,
q6 near-teacher.

Ready for Noah at 09:00: VQ rung geometry choices (all machinery
validated; leverage probe code-complete via peer's 55864f9; 128GB-tier
target would be the first data-free GLM-5.3 to fit a single consumer
box — q3 affine at 129 GiB does NOT fit one and is the worst row).

## 2026-08-30 (evening) — d8/K16384 fit resumed; rung arithmetic priced

Fit RESUMED from the 11 banked shards (all skipped instantly, resume
path healthy) and ran clean at ~1900s/shard, relerr steady ~0.352 against
a 0.45 abort. Script: scratchpad/glm_fit_resume.sh (glm5vlm venv,
--vq-layers 3-45 --k 16384 --dim 8 --family glm5_next).

VISION PROVISIONAL CLEARED: all 347 vision tensors in the struct base are
non-zero (|mean| 0.0023 to 1.625, sane). READINESS's "graft_vision is NOT
needed for glm5_next" is now MEASURED, not provisional. GLM assembly is
therefore pack -> bundle -> check -> smoke -> score: no splice-ple/
pack-ple (no PLE in this family) and no graft. Staged as
scratchpad/glm_assembly.sh.

RUNG ARITHMETIC (measured header pass on the struct base, 98.55 GiB):
  experts (VQ'd)      756 tensors  89.64 GiB   2-bit markers
  attention          1351          5.82        8-bit affine
  embed/lm_head        72          1.44        8-bit affine
  vision              347          1.05        bf16
  norms/router        472          0.61        bf16
Protected (non-expert) mass is only 8.91 GiB, so raising ALL of it from
8-bit to bf16 buys just +7.3 GiB -> ~106 GiB. CONCLUSION: the protected-
set lever CANNOT reach the 120 or 140 rungs; expert mass (89.6 of 98.6
GiB) is the only dial big enough, so both higher rungs need their OWN
expert fits at richer geometry (lower d and/or different K). Those should
be CHEAPER than this run, since K dominates fit cost (Flash-Next: K256
~9 min vs K4096 ~80). 120/140 geometry is Noah's call and is NOT yet on
record -- it is the only thing blocking an overnight queue.

RUNTIME IS NOT A BLOCKER (unlike MTP): glm5_next ships in RELEASED
mlx-vlm 0.6.17 (pip, not a local edit), so published GLM VQ artifacts
land on a runtime users can install today.
