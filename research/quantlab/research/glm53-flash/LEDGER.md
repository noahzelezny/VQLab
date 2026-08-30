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
