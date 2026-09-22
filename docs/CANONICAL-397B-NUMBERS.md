# Canonical 397B numbers — single source of truth for cards + paper revision
# Measured 2026-09-20-21 on CORRECTED (uniform, layers 0-59) weights.
# KL: vqlab kl-ladder, paired, 3 house corpora @ 12288, teacher_caches_397b,
#     same cache + positions for every row (spicyneuron scored on it, F170).
# Sizes: GiB (bytes / 2^30), following symlinks.
#   text     = index-file safetensors, EXCL vision tower and MTP sidecar (HEADLINE)
#   +tower   = text + bf16 vision graft (0.85 GiB; mlx-lm ignores it)
#   download = text + tower + optional mtp-head-q6.safetensors (5.4 GiB) = full snapshot
# Every rung ALSO ships the vision tower AND the MTP draft head. State both on the card.

build     text   +tower  download   prose    code     lit    mean   top-1
2.2 v2   100.0   100.9    106.3    276.4    98.6   183.1   186.0   84.8%
2.4      108.0   108.8    114.2    230.4    87.7   134.1   150.7   86.2%
2.6      119.2   120.1    125.5    164.1    58.1    60.5    94.2   88.5%
3.1      141.7   142.6    148.0     91.4    32.3    16.3    46.7   91.7%
spicy26  120.6     --       --     332.1    98.4   241.0   223.8   83.5%  (affine 2.6bit, text-only, no tower)

# Paper-only, NOT shipped to main:
2.2 v1        100.1  (flat d4/K128, published Aug rev 455463516501): prose 337.2 code 119.0 lit 261.4
2.2 v1-flat    96.7  (repaired, F169):                              prose 350.2 code 125.7 lit 276.0

# Like-for-like vs spicyneuron (text-to-text, both no tower):
#   2.2 v2 100.0 GiB  beats spicy 120.6 on prose(-17%)+lit(-24%), ties code  -> 20.6 GiB lighter
#   2.4    108.0 GiB  beats spicy on all three                                -> 12.6 GiB lighter
#   2.6    119.2 GiB  beats spicy on all three                                ->  1.4 GiB lighter
#   3.1    141.7 GiB  beats spicy 2.6bit on all three (and 3.5bit @165.6)
#   crossover: uniform VQ overtakes affine just under 2.4; mixed VQ (2.2 v2) already ahead at 100 GiB

# top-1 agreement, per corpus (prose/code/lit) + mean. Card's top-1 column = PROSE.
# We beat spicyneuron on top-1 on EVERY corpus and on the mean -- KL and top-1 agree.
#   2.2 v2  84.8 / 94.4 / 93.5  (mean 90.9)
#   2.4     86.2 / 94.9 / 95.3  (mean 92.1)
#   2.6     88.5 / 95.9 / 97.6  (mean 94.0)
#   3.1     91.7 / 97.0 / 99.3  (mean 96.0)
#   spicy   83.5 / 94.0 / 91.7  (mean 89.8)
