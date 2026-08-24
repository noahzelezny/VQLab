# REPRODUCING — paper table rows → commands

Every row in the paper is reproducible in the *statistical* sense: every
artifact the paper measures is a single UNSEEDED draw, so a re-fit
reproduces aggregate scores to reporting precision, not exact bytes (paper §5; two artifacts of identical geometry differ, and
the measured fit-to-fit spread is the noise floor every margin is read
against). Note the asymmetry with what you will run today: **dense fits are
now seeded by default**, so your repeat of a dense recipe reproduces far
more tightly than the paper's artifacts did — though still not bitwise, and
that difference matters when measuring floors (see below). MoE fits remain
unseeded draws. One historical exception is stated at the end.

Conventions below: `<BF16>` = the family's bf16 source directory; `<BASE>` =
the affine skeleton (397B `struct6-tail3x3`); all outputs go to NEW
directories — never refit into an existing artifact's path (fits resume,
and an in-place refit destroys the evidence for any published number).

## The full pipeline, once (397B flat K256 as the example)

```bash
# 0. price it first (optional but free)
vqlab price --family qwen397b --budget-gib 112

# 1. fit
vqlab fit-moe --base <BASE> --src <BF16> --out fits/K256 \
    --vq-layers 0-56 --k 256 --dim 4 --family qwen3_5

# 2. outlier gate — on a box that did not produce the fit
vqlab verify --artifact fits/K256 --src <BF16> --family qwen3_5 --outlier 3.0

# 3. pack to true bit-width (recomputes index total_size from packed shards)
vqlab pack --src fits/K256 --out artifacts/K256-packed

# 4. graft the vision tower (post-graft size is the citable size)
vqlab graft --artifact artifacts/K256-packed --src <BF16>

# 5. release + bundle gates
vqlab check-release --artifact artifacts/K256-packed
vqlab check-bundle --artifact artifacts/K256-packed
vqlab bundle-accept artifacts/K256-packed   # tests the copy that ships

# 6. smoke: one generated token through the shipping fused path
vqlab smoke artifacts/K256-packed   # preflights RAM, then generates and
                                    # NAMES the runtime that actually resolved

# 7. score, then stamp
vqlab score --model artifacts/K256-packed                        # prose
vqlab score --model artifacts/K256-packed \
    --corpus src/vqlab/referee/referee_corpus_code.txt           # code
vqlab manifest write artifacts/K256-packed
```

## §3.1 — the 397B ladder (streaming referee ppl, prose/code)

| row | fit command (steps 2–7 identical) |
|---|---|
| d8/K16384, 100.97 GiB | `vqlab fit-moe ... --k 16384 --dim 8` |
| flat K128, 100.93 | `... --k 128 --dim 4` |
| harvest K64/K256, 107.9 | `... --k 256 --dim 4 --geom` shallow-band K64 (see `fit-moe --help`, `--tail-geom`/`--vq-layers` split) |
| flat K256, 111.62 | `... --k 256 --dim 4` |
| flat K512, 122.31 | `... --k 512 --dim 4` |
| harvest K512/K2048, 139.93 | K2048 body, K512 shallow band (L0–9) |
| flat K2048, 143.68 | `... --k 2048 --dim 4` |

Notes: d8/K16384 runs the device-memory codebook kernels (its 256 KiB
codebook is 8x over the threadgroup cap — safe by design, not by cap
arithmetic). Healthy relerr scales with K: set `--relerr-abort` per
geometry (~0.19 K2048-class, ~0.31 K256, ~0.46 K128). The 143.68 GiB
flagship cannot be smoked on a 96 GB box (preflight will refuse).

Comparator rows (spicyneuron 2.6/3.5-bit) are community artifacts scored
on the same referee, same session, after `vqlab check-comparator`.

## §3.2 — 35B MoE row (KL-to-bf16)

```bash
vqlab kl cache --model <BF16-35B> --out-dir caches/qwen36 \
    --num-samples 128 --seq-len 512 --top-k 64
vqlab fit-moe --base <BASE-35B> --src <BF16-35B> --out fits/35b-K8192 \
    --k 8192 --dim 4 --family qwen3_5 --vq-layers 0-39
# gate, pack, smoke as above, then:
vqlab kl score --model artifacts/35b-K8192-packed --cache-dir caches/qwen36
```

Published measurement: 53.022 mnats / 89.55% top-1 at 14.838 GiB packed —
reproduced to every printed digit through a fixed runtime, gate PASS,
bundled runtime tested as the unit under test.

## §3.2 — dense 27B ladder (KL + ppl)

```bash
vqlab kl cache --model <BF16-27B> --out-dir caches/qwen38 ...
vqlab fit-dense --src <SRC-27B> --out fits/27b-d2K512 \
    --k 512 --dim 2 --family qwen3_8_dense --relerr-abort 0.6
# splice the VQ fits into the affine base -> a RUNNABLE artifact.
# fit-dense alone is NOT a model: it emits only the VQ'd MLP tensors.
# --dry-run first: it asserts every fitted module lands on a real base
# module BEFORE any bytes are written (the name conventions differ).
vqlab build-dense --family qwen3_8_dense --base <q4-base> \
    --mlp fits/27b-d2K512 --out assembled/27b-d2K512 --dry-run
vqlab build-dense --family qwen3_8_dense --base <q4-base> \
    --mlp fits/27b-d2K512 --out assembled/27b-d2K512

vqlab pack-dense --src assembled/27b-d2K512 --out artifacts/27b-d2K512-packed
vqlab check-bundle --artifact artifacts/27b-d2K512-packed
vqlab verify --artifact artifacts/27b-d2K512-packed --src <BF16-27B> \
    --family qwen3_8_dense --outlier 3.0
vqlab kl score --model artifacts/27b-d2K512-packed --cache-dir caches/qwen38
vqlab score --model artifacts/27b-d2K512-packed   # ppl leg
```

Affine comparators q2–q8 are *local* mlx-lm conversions (stated in the
paper); build them with stock `mlx_lm.convert` and re-verify with
`check-comparator` before citing their rows.

## §2.6 — noise floors

n≥3 fits of the same recipe into three NEW directories, score all, quote
the range. **The dense fitter seeds by default (`--seed 1234`), and a floor
measured at a fixed seed is not a floor** — it would certify every
third-decimal margin as real. A floor needs independent draws, so either
unseed or vary the seed:

```bash
for i in 1 2 3; do
  # --seed -1 restores unseeded behaviour: a fresh draw each time.
  # (Equivalently: --seed $i, three distinct seeds.)
  vqlab fit-dense --src <SRC-27B> --out fits/floor-$i \
      --k 256 --dim 2 --family qwen3_8_dense --seed -1
  # pack + score each
done
```

The MoE fitter takes no `--seed` at all — every MoE fit is a fresh draw,
so the loop above is correct there as written.

**What a mistakenly-seeded floor looks like:** not zero. Seeded fits are
near-identical but not bitwise identical — 0.0100% of codes still differ
with the RNG pinned (measured at d2/K256), because a second nondeterminism
source survives the seed. So the mistake returns a small, nonzero,
plausible-looking number rather than an obviously broken one, and it will
be far tighter than any real floor at that geometry. Treat a surprisingly
tight floor as a symptom of a pinned seed, not as evidence your fits are
unusually consistent.

Measured: dense 27B d2/K256 → KL range 2.085 mnats, ppl 0.0447 (n=3);
397B d4/K256 → 0.0256 prose / ~0.0178 code (n=2, same-stack); 397B d4/K2048
→ 0.0056 prose / 0.0104 code (n=2). Floors NARROW as K grows, which is why
inheriting one across geometries is forbidden. (An earlier inferred pair,
0.0134/0.0161, is superseded — margins computed against it read about 2x
too favourable.) A margin inside
its floor is noise, and the paper reports three of its own margins that way.

## §3.4 — speed

Same-session ratios only, n≥3 per arm, one process per box, model resident
(`preflight-ram` first). Decode at ~100 GiB residency is bimodal on our
hardware; absolutes are unsafe and no absolute is published.

## What cannot be reproduced, and why we say so

The originally-published 397B 2.4bpw build predates the manifest system and
its base was later overwritten in place; it is preserved and checkable (its
manifest and pinned revision exist) but not rebuildable. Every other row is
a fit-and-score away. This incident is why `vqlab manifest` exists — stamp
every artifact you gate, and `manifest check` before citing it.
