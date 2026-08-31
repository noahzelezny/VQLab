# VQLab

**Custom size-targeted vector-quantized (VQ) builds of large models on Apple
Silicon, with MLX.** Name a byte budget, price the recipe before fitting,
fit it data-free, pack to true bit-width, verify, and serve through stock
`mlx-lm` — the runtime ships inside the artifact.

The method and every number below are documented in the companion paper,
*Data-Free Vector Quantization Beats Affine Quantization at Matched Bytes
Below 6 Bits* — doi:[10.5281/zenodo.22119017](https://doi.org/10.5281/zenodo.22119017). Released artifacts are published
under [TheDrainFlorist](https://huggingface.co/TheDrainFlorist) on Hugging
Face with their VQ runtimes bundled in-checkpoint (stock `mlx-lm`, no
patches).

## Published models

Every artifact this tool's pipeline has shipped:

| model | family |
|---|---|
| [Qwen3.5-397B-A17B-VQ-2.2bpw](https://huggingface.co/TheDrainFlorist/Qwen3.5-397B-A17B-VQ-2.2bpw) | 397B MoE (d8/K16384) |
| [Qwen3.5-397B-A17B-VQ-2.4bpw](https://huggingface.co/TheDrainFlorist/Qwen3.5-397B-A17B-VQ-2.4bpw) | 397B MoE |
| [Qwen3.5-397B-A17B-VQ-2.6bpw](https://huggingface.co/TheDrainFlorist/Qwen3.5-397B-A17B-VQ-2.6bpw) | 397B MoE |
| [Qwen3.5-397B-A17B-VQ-3.1bpw](https://huggingface.co/TheDrainFlorist/Qwen3.5-397B-A17B-VQ-3.1bpw) | 397B MoE (flagship) |
| [Qwen3.6-35B-A3B-VQ-3.4bpw](https://huggingface.co/TheDrainFlorist/Qwen3.6-35B-A3B-VQ-3.4bpw) | 35B MoE |
| [Qwen3.6-35B-A3B-VQ-3.8bpw](https://huggingface.co/TheDrainFlorist/Qwen3.6-35B-A3B-VQ-3.8bpw) | 35B MoE |
| [Qwen3.6-35B-A3B-VQ-4.6bpw](https://huggingface.co/TheDrainFlorist/Qwen3.6-35B-A3B-VQ-4.6bpw) | 35B MoE |
| [Qwen3.6-35B-A3B-VQ-5.4bpw](https://huggingface.co/TheDrainFlorist/Qwen3.6-35B-A3B-VQ-5.4bpw) | 35B MoE |
| [Qwen3.8-27B-VQ-3.9bpw](https://huggingface.co/TheDrainFlorist/Qwen3.8-27B-VQ-3.9bpw) | dense 27B |
| [Qwen3.8-27B-VQ-4.5bpw](https://huggingface.co/TheDrainFlorist/Qwen3.8-27B-VQ-4.5bpw) | dense 27B |
| [Qwen3.8-27B-VQ-4.8bpw](https://huggingface.co/TheDrainFlorist/Qwen3.8-27B-VQ-4.8bpw) | dense 27B |
| [gemma-4-26b-a4b-it-VQ-6.2bpw](https://huggingface.co/TheDrainFlorist/gemma-4-26b-a4b-it-VQ-6.2bpw) | gemma MoE * |
| [gemma-4-e4b-it-VQ-PLE](https://huggingface.co/TheDrainFlorist/gemma-4-e4b-it-VQ-PLE) | gemma dense * |

\* Released and usable, but no quality claims are made for the gemma family
anywhere in this repo or the paper: its scoring instrument is
non-deterministic (see "Known scope limits"). All sizes are measured packed bytes; every margin is stated against
a measured seed-noise floor; see [METHODOLOGY.md](METHODOLOGY.md) for the
rules that keep these numbers honest.

## Measured results

**397B MoE (Qwen3.5-397B-A17B), streaming-referee perplexity, prose/code:**

| build | GiB | prose ppl | code ppl |
|---|---|---|---|
| d8/K16384 | 100.97 | 3.0591 | 2.6728 |
| flat K256 (2.4bpw) | 111.62 | 2.7655 | 2.6383 |
| **flat K512** | **122.31** | **2.5634** | **2.6123** |
| flat K2048 (flagship) | 143.68 | 2.3410 | 2.5963 |
| calibrated 2.6-bit comparator | 120.6 | 3.1843 | 2.6667 |
| calibrated 3.5-bit comparator | 165.6 | 2.3614 | 2.6005 |

- **flat K512 beats the calibrated 2.6-bit build by 0.6209 prose ppl (24x the
  K256 floor) and by 0.0544 on code (3.1x), at +1.7 GiB** — the cleanest
  like-for-like on the ladder.
- **d8/K16384 beats the same comparator while being 19.6 GiB smaller**
  (prose margin 0.1252 = 4.9x the K256 floor). No d8 floor has been
  measured, so that multiple is *borrowed* and reads as a lower bound on
  confidence, not a measurement. It is the ~101 GiB build a 128 GB Mac holds.
- **The flagship is 21.9 GiB smaller than the calibrated 3.5-bit build**, with
  prose better by 0.0204 = 3.6x the K2048 floor (claimed) and code a tie
  (0.4x, inside the floor). "Wins both corpora" is withdrawn.

*Reproducibility note:* prose-ppl rows are fully reproducible from the
shipped corpus. The **code-ppl column was measured on a private corpus that
does not ship**; `scripts/make_code_corpus.py` builds a public replacement
with a provenance manifest, but that is a different instrument and its
scores do not compare to the column above.

**Read the floors with the margins.** Seed-noise floors are geometry-specific
and narrow as K grows: 397B d4/K256 = 0.0256 prose / ~0.0178 code; d4/K2048 =
0.0056 / 0.0104; dense 27B d2/K256 = 2.085 mnats / 0.0447 ppl. A margin
inside its floor is not a claim, and a floor borrowed from another geometry
is labelled as borrowed.

**Size basis, disclosed rather than restated.** Our 397B builds carry the
bf16 vision tower (0.8494 GiB); both calibrated comparators are text-only.
Every 397B size advantage above is therefore *understated* by roughly that
much. We keep the download-size convention and disclose the offset, because
restating sizes would move every number in our own favour.

**35B MoE (Qwen3.6-35B-A3B), KL-to-bf16 — quality only:** our d4/K8192
build measures **53.022 mnats KL / 89.55% top-1**, against the community
4-bit's 78.557 / 85.61% on the same instrument — a **32% KL reduction**.
The packed artifact passed the outlier gate on a second box, generated
through its shipping kernel, and reproduced its score to every printed
digit.

*No size comparison is claimed for this pair.* Our 35B rungs are text-only
while the community quants carry a 333-tensor bf16 vision tower
(0.832 GiB measured), so the two sizes are computed on mismatched bases
and the delta moves once restated. Our artifact is 14.838 GiB packed,
measured; the comparison to the incumbent's footprint is deferred rather
than quoted on a basis we know to be mismatched.

**Dense 27B (KL-to-bf16 + ppl):** the recipe is not an MoE phenomenon.
d2/K512 (the published artifact: 32.81 mnats / 90.84% top-1 / 5.162 ppl)
beats the 4-bit affine conversion by 28.4% KL (6.2x floor) and +1.02 pp
top-1 at 4-bit-class size; d4/K1024 beats q3 on both metrics at
0.35 GiB less.

**Where the method ends, measured:** the VQ advantage lives at the low-bpw
end — wins measured at 2.0–4.5 bpw across three families; the VQ/affine
crossover is bracketed at 4.5–6.0 bpw on the dense 27B; by 8 bits affine is
essentially lossless and the advantage is gone. Prefill remains ~0.5x affine
at 35B. We publish these fences as measured boundaries, not caveats.

## The differentiating feature: size targeting

Flat rungs leave gaps. VQLab prices an artifact **before fitting it**:

```bash
vqlab price --family qwen397b --budget-gib 108
```

Two measured size models back this: the 397B harvest form
(`new = base − 1.87 GiB × shallow_bits_harvested`, 6-for-7 within ±0.4 GiB)
and the dense composition form (`total = codes + scales + carry`, closed to
≤0.003 GiB across three builds and two geometries). Harvesting shallow-layer
K back is ~2x the byte-efficiency of stepping down a flat rung — it buys the
sizes between rungs. It is never free and never beats a flat rung at the flat
rung's own size; the pricer tells you both.

## Pipeline

```
fit → verify (outlier gate) → pack → graft (vision) → release checks
    → smoke-generate through the shipped kernel → score
```

```bash
# Install into a DISPOSABLE venv, never a shared/base interpreter: this
# package ships a model runtime, and which copy of a runtime resolves is a
# real source of wrong conclusions (METHODOLOGY.md §5).
python3 -m venv .venv && . .venv/bin/activate
pip install -e .
vqlab selftest        # real pipeline on a tiny synthetic model (<1 min, uses the GPU)

# MoE families (Qwen3.5/3.6-class): fit against an affine skeleton + bf16 source
vqlab fit-moe --family qwen3_5 --base <affine-skeleton> --src <bf16> \
    --vq-layers 0-56 --k 256 --dim 4 --out fits/K256
vqlab pack  --src fits/K256 --out artifacts/K256-packed
vqlab graft --artifact artifacts/K256-packed --src <bf16>   # vision tower

# Dense families: fit the MLP trio, splice into a quantized base
vqlab fit-dense --family qwen3_8_dense --src <bf16> --k 512 --dim 2 \
    --out fits/d2K512
vqlab build-dense --family qwen3_8_dense --base <q4-base> \
    --mlp fits/d2K512 --out assembled --dry-run   # then without --dry-run
vqlab pack-dense --src assembled --out artifacts/d2K512-packed

# Gates — run these before believing anything (see METHODOLOGY.md)
vqlab verify --artifact fits/K256 --src <bf16> --family qwen3_5 \
    --outlier 3.0                       # BEFORE any score is believed
vqlab check artifacts/K256-packed       # release + bundle gates
vqlab smoke artifacts/K256-packed       # generate through the SHIPPED runtime

# Score (referee ppl streams > RAM; KL needs a teacher cache)
vqlab score --model artifacts/K256-packed
vqlab kl cache --model <bf16> --out-dir caches/family
vqlab kl score --model artifacts/K256-packed --cache-dir caches/family

vqlab manifest write artifacts/K256-packed   # stamp provenance
```

Every subcommand is a thin wrapper over a standalone script in
`src/vqlab/`; `vqlab <cmd> --help` shows the full surface. The dense block
above is exactly the sequence that was dogfooded end-to-end on a real model
in a fresh venv (REPRODUCING.md records it, and maps each paper table to
its commands).

## MTP speculative decoding

VQLab also ships a decode strategy: **multi-token-prediction speculative
decoding** for any mlx-lm model that has an MTP drafting head. It is a
library, not a server and not a fork — mlx-lm loads the model and owns the
architecture; we replace only the decode loop.

```python
from mlx_lm import load
from vqlab import load_mtp_head, mtp_generate, mtp_stream_generate

model, tok = load(path, trust_remote_code=True)
head, _ = load_mtp_head(model, model_path=path)     # or sidecar=<file>

print(mtp_generate(model, tok, "Explain VQ.", head, temp=0.7, top_p=0.9))

for r in mtp_stream_generate(model, tok, "Explain VQ.", head, max_tokens=256):
    print(r.text, end="")                            # r.acceptance, r.steps, …
```

```bash
vqlab mtp-pack     --model <artifact> --mtp <graft.safetensors>  # build sidecar
vqlab mtp-generate --model <artifact> --temp 0.7
vqlab mtp-bench    --model <artifact> --tokens 128               # the numbers
```

The head drafts token t+2 from (trunk hidden at t, embedding of t+1), so each
step verifies one speculative token inside a single 2-token trunk forward:
accepted gives two tokens for one forward; rejected rolls the caches back and
replays, costing one extra forward and never a wrong token.

### Measured (Qwen3.8-Flash-Next, 6-bit head, greedy, 96 tokens)

> The table below is M3 Ultra. The 2026-08-31 measurements below it are M4
> Max over SMB — different core counts, bandwidth and thermals, so speedups
> are **not** comparable across the two. Only within-run comparisons are.

| rung   | head  | baseline   | speculative | speedup | acceptance |
|--------|-------|-----------|-------------|---------|------------|
| 2.1bpw | 6-bit | 16.07 t/s | 26.87 t/s   | 1.67x   | 0.708      |
| 3.2bpw | 6-bit | 15.19 t/s | 23.71 t/s   | 1.56x   | 0.625      |

Across runs: **1.56–1.80x (median 1.65x)**, acceptance 0.578–0.812 (median
0.679). The 6-bit head is **2.12 GiB resident**, measured as an
`mx.get_active_memory` delta. Sidecars are named outside mlx-lm's
`model*.safetensors` glob, so a model directory carrying one still loads
normally through the stock loader — the head is optional residency.

### Head-cache alignment: measured neutral (2026-08-31)

`qwen4_exp` reads the head's rotary positions straight off its cache offset,
so the old loop fed the head positions compressed 2x and drifting with
generation length, rolled its cache back on every rejection, and never seeded
it over the prompt. `align="committed"` fixes all three. On a 512-token
greedy run at the 2.1bpw rung (M4 Max, 256 steps, one model load):

| head | align | acceptance | tok/s | speedup |
|------|-------|-----------|-------|---------|
| q6 (2.12 GiB) | committed | 0.7812 ±5.1pp | 29.83 | 1.566x |
| q6 (2.12 GiB) | legacy    | 0.7539 ±5.3pp | 29.43 | 1.545x |
| experts q4 / rest q8 (1.55 GiB) | committed | 0.7773 ±5.1pp | 29.52 | 1.551x |
| experts q3 / rest q8 (1.25 GiB) | committed | 0.7695 ±5.2pp | 29.31† | 1.548x† |

**The alignment fix buys no measurable acceptance: +2.7pp at 0.73σ.** It was
predicted to matter and it does not, at this generation length. It is kept as
the default because it is mechanically correct, costs nothing measurable
(+0.4pp speed, 0.06σ), and removes the draft-cache rollback entirely — not
because it was shown to help. An earlier 96-token run showed +12.5pp, which
was 1.53σ and did not survive the powered run; it is recorded here as a
falsified prediction, per METHODOLOGY.

The untested part of the hypothesis is length: legacy's position error grows
with generation, so any effect should widen well beyond 512 tokens.

**Timing on this machine is not trustworthy and the speedups above should be
treated as indicative only.** The M4 Max is a laptop, and a re-run of the
identical configuration in the same process gave 1.723x and 1.135x — a 25%
swing from thermal drift alone. The baseline pass is longer and hotter than
the speculative one, so it degrades first and *inflates* the ratio early. A
palindromic run order does NOT fix this (it pins whichever configuration sits
in the middle to both hottest slots — that error is mine, and it depressed the
legacy row). Publishable speedups need a thermally stable machine.

Acceptance, by contrast, is completely reliable here: greedy decode is
deterministic and every figure above reproduced to four decimals across
repeats. Note the corollary — repeating a greedy run yields **no** new
statistical information, and treating one trajectory's 256 steps as 256
independent trials overstates the power, because the steps share a prefix. The
alignment question needs independent *prompts*, not repeats.

† e3q8's speed is taken from an early, thermally clean slot; an earlier reading
of 22.36 tok/s was purely its position in the sequence.

### Mixed-bit heads

The 512-expert MoE stack is 4.688 of the head's 4.856 GiB — **96.5%, in two
tensors** — so head size is essentially one dial, and protecting the other
3.5% at a high bit-width is nearly free (+0.02 GiB from 6- to 8-bit across
all of it). `vqlab mtp-pack --expert-bits` sets the experts independently.

**Experts at 4-bit with everything else at 8-bit is 1.55 GiB against 2.12,
for no measurable acceptance cost (−0.4pp, 0.11σ) and no measurable speed
cost.** That is 27% of the head's residency recovered for nothing, which
matters most on the large rungs where headroom is tight. 3-bit experts hold
acceptance too (−1.2pp, 0.32σ) but need a clean re-measure for speed.

This search is safe by construction: head precision **cannot** affect output
quality, because the trunk verifies every drafted token. A coarser head costs
a rejection, never a wrong token.

Four things that are settled by measurement and should not be "simplified":

- **The head's cache offset is its position signal.** `qwen4_exp` reads rotary
  positions straight off it (`Attention.__call__`: `offset = cache.offset`).
  Keep one head row per *committed* token — seed over the prompt, advance two
  positions per step after verification — and the offset is the true position
  by construction. Advancing it once per step (two tokens) compresses positions
  2x and drifts further every step. MTPLX documents the same invariant for this
  architecture and reaches it by overriding the rope offset explicitly.

- **The head's norms must be applied by the architecture's own RMSNorm.**
  qwen4_exp's is zero-centered (`y = norm(x) * (1 + weight)`); hand-rolling
  `n * w` drops the `+1.0` and drives acceptance to exactly **0.0**. That one
  mistake was the entire original bug. See `src/vqlab/mtp_head.py`.
- **Attention caches roll back by the offset DELTA, not a fixed count.**
  Trimming a hardcoded 1 leaves a stale key while the recurrent caches roll
  back 2, and the two streams then drift silently.
- **Head precision cannot affect output quality.** The trunk verifies every
  drafted token, so a worse draft costs a rejection, never a wrong token;
  6-bit and bf16 measured identical acceptance (0.7065 vs 0.7051 at n=4096,
  inside the ±0.7pp binomial SE).

At temperature the acceptance test is exact rejection sampling with residual
correction (Leviathan et al. 2023; Chen et al. 2023), so the output
distribution is the one plain sampling from the trunk would give, for **any**
draft quality. Samplers are mlx-lm's own (`sample_utils`), adapted to return
the distribution alongside the draw. `tests/test_mtp_sampling.py` checks the
preservation claim both in closed form and empirically.

**Bit-identical output against single-token decoding is not achievable and is
not a valid correctness gate.** MLX's chunked and single-token kernels
disagree at genuine near-ties (measured top-2 logit gaps of 0.25 and exactly
0.00 against a median of 3.625), and verification always happens inside a
2-token forward. The gate is *divergence confined to near-ties*, which
`vqlab mtp-bench` measures directly as its chunk control.

### Can a VQ artifact use a native MTP runtime? Not today (2026-08-31)

oMLX 0.6.3+ serves `qwen4_exp` with its own native MTP ("Lightning MTP"), so
the obvious question is whether a VQLab VQ rung can borrow it. Tested directly
against oMLX 0.6.4. It cannot, and **the blocker has nothing to do with the
MTP head**:

- oMLX **does** honour `model_file` — but only on its mlx-lm path
  (`omlx/patches/deepseek_v4/utils_patch.py`). Its `qwen4_exp` is vendored into
  **mlx-vlm**'s namespace (`omlx/patches/mlx_vlm_qwen4_exp_compat/`), and the
  mlx-vlm loader does not honour `model_file`.
- Its `qwen4_exp` expects **mlx-vlm key layout** — `language_model.*`,
  `vision_tower.*`. VQLab artifacts are **mlx-lm layout** — `model.*`,
  `lm_head.*`, `model.visual.*`. A lazy load gets all the way through
  architecture construction and then rejects all 3671 tensors as "not in
  model".

So two changes are needed together, and only one is ours: emit VQ artifacts in
mlx-vlm layout, **and** have the mlx-vlm path honour `model_file` — without
which the VQ modules cannot be constructed at all and the packed codes have
nothing to decode them. The second is an upstream feature request, and oMLX
already implements exactly that for its mlx-lm path.

What DOES work with oMLX today is `vqlab mtp-graft` output on a stock
(non-VQ) Flash-Next checkpoint, since its `Qwen4ExpMTPModule` accepts `mtp.`,
`language_model.mtp.`, `model.mtp.` and `model.language_model.mtp.` prefixes.
That is not a VQLab differentiator — Qwen's own head serves the same purpose —
but it is the reason `mtp-graft` gates on key-set parity rather than guessing.

### Adding a family

A family is a `FamilySpec` table entry in `src/vqlab/mtp/registry.py` plus a
head module — nothing else in the package names an architecture. The entry
says where the head lives, which submodule's input is the pre-lm_head
activation, which cache the head uses, and whether the family's recurrent
caches may use free (non-copying) snapshots. That last field defaults to the
safe `"copy"`: qwen4_exp's free snapshots work because it *reassigns* cache
slots rather than mutating them, which is an implementation accident, not a
contract. `caches.check_snapshot_semantics` is the measurement that earns a
family the cheap path.

Registered: `qwen4_exp`. GLM-5.3 and DeepSeek also ship MTP heads and the
registry is shaped for them, but neither is registered here because neither
can be tested in this repo today — a table entry without a measured
acceptance number is not evidence of anything.

## Requirements

- Apple Silicon Mac; RAM sized to the artifact for fit/smoke (scoring
  streams and can exceed RAM). The 397B fits used 96–128 GB machines.
- Python ≥ 3.12, `mlx`, `mlx-lm` (stock — artifacts bundle their own
  runtime), `numpy`, `safetensors`.

## Verifying your install

**It does real GPU work.** The fits and kernels are genuine, so although it
takes seconds of GPU, it *contends* — do not run it on a machine that is
mid-experiment.

`vqlab selftest` is not a mock: it synthesizes a small checkpoint and runs
the shipped fitter, outlier gate, packer, manifest and Metal kernels over it
as subprocesses, checking what each stage is supposed to guarantee — seeded
fits reproduce, packing is bit-exact, the gate fails a collapsed tensor, the
manifest catches altered bytes, and a dense bundle serves on a stock mlx-lm.
Every gate is exercised in **both** directions, because a gate that only ever
passes tells you nothing.

`pytest` covers the MTP decode strategy on a synthetic model — rollback,
distribution preservation, and the cache-snapshot gate, each in both
directions. `tests/test_mtp_integration.py` runs the same checks against a
real artifact and is skipped unless `VQLAB_MTP_MODEL` points at one. The two stages that need a real multi-GB
checkpoint — end-to-end generation and scoring — are reported as SKIPPED with
the reason, never silently dropped.

## Honesty rules baked into the tool

- The outlier gate refuses to be skipped quietly: scoring an ungated
  artifact is on you, and the docs say what it cost us.
- Sizes are computed from packed shards on disk, never from index metadata.
- The fitter aborts on per-geometry relative-error thresholds and never
  writes in place (fits resume).
- `bundle-accept` tests the runtime copy *lifted from the artifact*, not
  whatever your import path resolves to.

## Known scope limits

- MLX/Metal only; kernel conclusions (threadgroup ceiling, d8 decode tax)
  are Apple Silicon specific.
- Families onboarded: Qwen3.5-397B-A17B, Qwen3.6-35B-A3B (MoE),
  dense Qwen 27B. Gemma-family fitting code is included, but no quality
  claims are made for it: its raw perplexity is invalid as a property of
  the model, which makes scoring non-deterministic.
- New family? Read `docs/ONBOARDING.md` (the two-hour characterisation pass
  to run before fitting anything).

## Development note

This repository was assembled with AI assistance (Claude Code). Direction,
review, measurement and every published claim are the author's; the tool
wrote code and prose under his review, and the artifacts and numbers are
the lab's. Recorded here as disclosure of process — a tool is not an author.

## License

Apache-2.0 for all code. The shipped corpus data files carry their own
terms (CC BY-SA for the WikiText prose corpus, public domain for the
literary corpus) — see [docs/CORPORA.md](docs/CORPORA.md) for per-file
provenance and attribution. If you use VQLab or its artifacts in published work, cite the
companion paper.
