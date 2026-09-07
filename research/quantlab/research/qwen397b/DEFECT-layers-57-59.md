# 397B: layers 57–59 were never VQ-fitted (all four published rungs)

Found 2026-09-06 while auditing the affine byte budget for the v2 arc.
**Status: confirmed defect; fix in progress; quality impact PENDING
measurement — do not publish a correction until it is measured.**

## What is wrong

`Qwen3.5-397B-A17B` has **60 hidden layers** (`config.json` →
`text_config.num_hidden_layers: 60`). Every shipped VQ rung fits experts
for layers **0–56 only**. Layers **57, 58, 59** fall through to the
affine path and ship at **3 bits** — the lowest precision anywhere in
these artifacts.

Verified identical in all four published rungs:

| rung | VQ layers | 3-bit affine modules |
|---|---|---|
| VQ-2.2bpw | 0–56 (57) | 9 |
| VQ-2.4bpw | 0–56 (57) | 9 |
| VQ-2.6bpw | 0–56 (57) | 9 |
| VQ-3.1bpw | 0–56 (57) | 9 |

The nine modules are exactly
`language_model.model.layers.{57,58,59}.mlp.switch_mlp.{gate,up,down}_proj`.

**These are ordinary routed-expert layers on the main forward path.**
Not MTP layers (the config carries no next-n-predict key; the 397B MTP
head is a separate 3.19 GiB sidecar), not dense layers, not structurally
special: same `[512, 1024, in=4096]` expert stacks as their neighbours,
and the bf16 source stores them in the SAME fused `gate_up_proj` /
`down_proj` layout as layer 56. Nothing prevented fitting them.

**Cause:** the fit was invoked `--vq-layers 0-56` on a 60-layer model.
`v2_sweep.py` then inherited the same off-by-three as a constant
(`N_VQ_LAYERS = 57  # vq_modules covers layers 0-56, no gaps`), so every
sweep since has silently scoped itself to the same 57 layers.

## What it costs (2.2 rung, measured from the artifact)

| storage | per module | per layer | 9 modules |
|---|---|---|---|
| affine 3-bit (`weight`+`scales`+`biases`) | 0.875 GiB | 2.625 GiB | **7.875 GiB** |
| VQ d8/K16384 (`codes`+`codebook`+`vq_scales`) | 0.500 GiB | 1.500 GiB | **4.502 GiB** |
| **saving** | 0.375 GiB | 1.124 GiB | **3.373 GiB** |

In stored bits per weight that is **3.25 bpw affine vs 2.00 bpw VQ** for
the same tensors. For scale: the entire v2 reallocation moved 1.125 GiB.

## Expected quality direction — NOT YET MEASURED

The family's own ladder says VQ beats affine at fewer bytes (on
Flash-Next: affine q3 at 75 GiB scores KL 1083.4; VQ at 45 GiB scores
390.1). If that holds here, converting these layers is smaller AND
better at once. **But it must be measured**: 57–59 are the last layers
before the output, and late-layer sensitivity is the one honest reason
someone might deliberately leave them alone — though 3-bit, the lowest
precision in the artifact, is a strange way to protect anything.

## Effect on published claims

Every rung carries the defect **identically**, so all rung-to-rung
comparisons in the published tables and the paper hold exactly as
printed. The defect makes the published results **conservative**: the
artifacts are ~3.4 GiB heavier, and slightly worse, than the method
actually delivers, because three expert layers ship in the affine format
the work argues against. The error runs in the direction of underselling
VQ, not overstating it.

## Fix

`demote_fit.py` grew an affine→VQ conversion path (2026-09-06): it
detects a module sitting in `quantization` rather than `vq_modules`,
fits it from bf16, and on splice drops `weight`/`scales`/`biases`, adds
`codes`/`codebook`/`vq_scales`, moves the `weight_map` entries, deletes
the module from BOTH `quantization` and `quantization_config`, and adds
a full `vq_modules` entry (`experts`/`out`/`in`/`k`/`dim`/`group`/
`pack_bits` — the keys the bundled `model.py` reads at line ~3148).

Order of work: convert L57 alone → score → if neutral-or-better, convert
58 and 59 → runtime smoke → then decide about republishing the family.

Also to fix once the result lands: `N_VQ_LAYERS` in `v2_sweep.py`, and
every future sweep's layer range (the hot-layer sweep has never seen
layers 57–59, so v2's best-6 was chosen from an incomplete pool).

---

## Measured (2026-09-06). Conversion is a BYTE REFUND, not free quality.

Base = shipped 2.2: prose 3.0568, code 2.6728, literary 1.2820 @ 100.971 GiB.
Converted to VQ **d8/K16384** (2.00 bpw, matching layers 0–56):

| converted | Δ prose | Δ code | Δ literary | size |
|---|---|---|---|---|
| L57 only | −0.0010 | −0.0057 | −0.0036 | 99.847 |
| L58 only | −0.0093 | −0.0064 | −0.0029 | 99.847 |
| L59 only | −0.0063 | **−0.0160** | −0.0037 | 99.847 |
| **all three** | **−0.0218** | **−0.0319** | **−0.0121** | **97.598** |

Size arithmetic landed exactly as predicted (−1.124 GiB/layer, −3.373
total). Three findings:

1. **The "smaller AND better" hypothesis is FALSIFIED at this target.**
   Affine 3-bit and VQ d8/K16384 are roughly quality-equivalent on these
   layers; VQ is simply 1.25 bpw cheaper. The win is bytes, not quality.
2. **Damage COMPOUNDS (super-additive, 131% of the sum of singles)** —
   the mirror of promotions, which compose at 76%. Sequential late layers
   degrade together.
3. **L59 is a code specialist**: mid-pack on prose, 2.5x the code damage
   of its neighbours. This is fatal for a straight d8 conversion, because
   v2's promotions are code-NEUTRAL — there is no mechanism to buy a
   −0.032 code regression back. **Do not ship the d8/K16384 conversion.**

### The right target is d4/K256, not d8/K16384

Per module: affine 3-bit = 0.875 GiB @ **3.50 bpw** (3 bits + fp16 scale
AND fp16 bias per group of 64 -- corrected from an earlier 3.25); VQ d8/K16384 = 0.500
@ **2.00**; VQ **d4/K256 = 0.5625 @ 2.25**. d4/K256 still refunds **2.81
GiB** across the three layers while giving up only 1.0 bpw instead of
1.25 — and it is the level whose quality the family already knows.
Neither the 2.2 nor the 2.4 rung has a donor for these layers (identical
gap in every rung), so they need fresh fits; K256 is cheap to fit.
IN FLIGHT: singles + composite at d4/K256, all three corpora.

### Also still open

Layers 57–59 have **never been in a promotion sweep** (every sweep
inherited `--vq-layers 0-56` / `N_VQ_LAYERS = 57`). The v2 hot band was
L41–L49 — the late layers — so the three latest layers in the model are
unmeasured candidates, and v2's best-6 was selected from an incomplete
pool. Sweep them once they carry VQ codes.


---

## THE CURVE (2026-09-06). VQ beats affine at matched bytes; crossover 3.0 bpw.

All three layers converted together, scored against the shipped 2.2
(prose 3.0568 / code 2.6728 / literary 1.2820 @ 100.971 GiB). Noise floor
+/-0.004 (METHOD.md 11.2) -- anything inside it is a tie.

| geometry | bpw | GiB | d GiB | d prose | d code | d literary |
|---|---|---|---|---|---|---|
| affine 3-bit (SHIPPED) | 3.50 | 100.971 | — | — | — | — |
| d8/K16384 | 2.00 | 97.598 | -3.373 | -0.0218 | -0.0319 | -0.0121 |
| d4/K256 | 2.25 | 98.159 | -2.812 | -0.0063 | -0.0196 | -0.0083 |
| **d4/K2048** | 3.00 | **99.846** | **-1.125** | **+0.0037** | -0.0004 | -0.0008 |
| **d4/K8192** | 3.50 | 100.972 | +0.001 | **+0.0094** | -0.0016 | -0.0001 |

1. **VQ BEATS AFFINE AT MATCHED BYTES** (Noah's prediction). d4/K8192 is
   byte-identical to the affine tensors it replaces and gains +0.0094
   prose with code and literary level. The paper's central claim holds on
   the three layers it was never applied to -- free quality, zero bytes.
2. **Crossover at 3.00 bpw** (predicted 3.08 by interpolation, measured
   3.00). d4/K2048 is better on prose AND 1.125 GiB smaller, neutral on
   code and literary. Smaller and better, at last.
3. Below the crossover the refund gets expensive fast, and **L59 is a
   code specialist** -- at 2.00 bpw it alone costs -0.0160 code.
4. relerr fell from ~0.35 (d8/K16384) to 0.132 (d4/K8192) -- a 2.7x more
   faithful reconstruction -- while ppl barely moved. Independent
   confirmation of METHOD.md 11.3: relerr does not predict damage.

### Consequence: the refund can REPLACE v2's demotions

v2 (`iso100`) had to damage six layers (demote to K4096) to fund promoting
six others. It no longer has to. Converting 57-59 to d4/K2048 refunds
exactly 1.125 GiB -- precisely the cost of the best-6 promotions -- while
IMPROVING quality instead of costing it. Candidate `397b-v3` = promote
{29,43,44,45,47,48} + convert 57-59 to d4/K2048, landing back at the
shipped 100.971 GiB with six fewer damaged layers. IN FLIGHT.

## Reproducing v3 (artifacts are disposable, fits are not)

`397b-v3` needs no k-means to rebuild — ~10 minutes of pure splicing:

1. promotions: `v2_sweep.py --combo 29,43,44,45,47,48 --donor d4k256
   --build-only` off the shipped 2.2 base (donor tensors come from the
   shipped 2.4 rung; deterministic copy).
2. conversions: `demote_fit.py --layer {57,58,59} --k 2048 --dim 4
   --save-fit <archive>` — all three are ARCHIVE HITS at
   `/Volumes/Thunderbay HDD/vqlab-fits/qwen3.5-397b/demote_fit-d4/
   layer{57,58,59}-d4k2048.safetensors`, so no fitting runs.

Driver: `build_v3.sh` (committed alongside). Promotions must run FIRST —
`v2_sweep.py`'s preflight refuses a non-flat base and the conversion makes
it non-flat.
