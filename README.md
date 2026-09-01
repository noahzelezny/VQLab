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

### Serving

```bash
vqlab serve --model <artifact-dir> [--sidecar mtp-head-e3q8.safetensors] --port 8080
```

An OpenAI-compatible endpoint at `http://127.0.0.1:8080/v1`. Without
`--sidecar` it serves the artifact with no drafting, which is the useful
default: a VQ artifact needs the codebook kernels in its own bundled
`model.py`, so the environment that runs it is not interchangeable, and
shipping a server we know loads our artifacts is most of the value.

It is an **adapter, not a server**. mlx-lm already ships a complete
OpenAI-compatible server and calls generation at exactly one site, so VQLab
borrows that surface — templates, streaming, stop sequences, request schema —
and replaces only the decode strategy. Four patch points, verified at startup;
`serve` refuses to start if any has moved, because a server that quietly stops
drafting still answers every request correctly and only looks slower.

Verified on the 2.1bpw rung: greedy and temperature+top_p to 256 tokens,
streaming SSE, natural stop, served acceptance 0.773–0.925 — matching the
offline `mtp-accept` sweep, which is what confirms the served loop is the
measured loop.

Known limits: single user, no continuous batching, no cross-request prefix
reuse (a reused prefix cache would mis-position the drafting head, so the loop
**raises** rather than decoding at wrong positions).

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

### Speedup, measured on a thermally stable machine (2026-08-31)

2.1bpw rung, 512 tokens greedy, M4 Max with active cooling, 12 runs in
randomised balanced blocks with a 25s cooldown, each keeping its own adjacent
baseline:

| head | align | acceptance | speedup | tok/s |
|------|-------|-----------|---------|-------|
| q6 (2.12 GiB)   | committed | 0.7812 | **1.591x** ±0.0006 | 29.99 |
| e4q8 (1.55 GiB) | committed | 0.7773 | 1.586x ±0.0017 | 29.90 |
| e3q8 (1.25 GiB) | committed | 0.7695 | 1.583x ±0.0081 | 29.85 |
| q6 (2.12 GiB)   | legacy    | 0.7539 | 1.566x ±0.0044 | 29.52 |

Baseline **18.85 tok/s**, spread 18.83–18.89 across all twelve runs — **0.32%**.
The same measurement without active cooling spread 13.98–19.04 (26%) and was
worthless. The baseline spread is the readout for whether a speedup number
from a laptop means anything; quote it alongside, or do not quote the speedup.

**All three heads are within 0.5% of each other**, so the 1.25 GiB head buys
its 0.87 GiB back for essentially nothing in speed as well as acceptance.

The alignment fix is worth **+1.58% wall-clock** (t=9.7). That is close to
the most it *could* be worth, and the reason is structural rather than a
defect: at depth 1 each step emits exactly two tokens regardless, so
acceptance only changes how often a rejection costs a replay. On this prompt
the delta was 2.7pp, whose ceiling is +2.24% even with a free head; the
measured +1.58% implies the head costs ~0.5 of a trunk forward, absorbing
about a third of the available gain. Alignment's value is in acceptance, which
is what a deeper draft would spend — not in wall-clock at depth 1.

### Head-cache alignment: worth +5.9pp acceptance (2026-08-31)

`qwen4_exp` reads the head's rotary positions straight off its cache offset
(`Attention.__call__`: `offset = cache.offset`). The old loop advanced that
cache once per step while two tokens committed, rolled it back on every
rejection, and never seeded it over the prompt. `align="committed"` keeps one
head row per committed token, which makes the offset the true position by
construction.

Measured with `vqlab mtp-accept`: 12 independent prompts, 256 tokens each,
every prompt run through every configuration so the comparison is paired
(2.1bpw, M4 Max, one model load).

| head | committed | legacy | paired delta | t | wins |
|------|-----------|--------|--------------|---|------|
| q6 (2.12 GiB)   | 0.8171 | 0.7578 | **+5.92pp** | 6.34 | 12/12 |
| e4q8 (1.55 GiB) | 0.8105 | 0.7598 | +5.08pp | 4.17 | 11/12 |
| e3q8 (1.25 GiB) | 0.8151 | 0.7643 | +5.08pp | 4.14 | 11/12 |

1536 steps per cell. The effect replicates independently across all three
heads, and the 12/12 sign test alone is p ~ 0.0002.

**A methodological warning, recorded because we walked into it.** The first
attempt compared a single prompt and read +12.5pp at n=48 steps; the second
read +2.7pp at n=256 and was written up here as "measured neutral, prediction
falsified". Both were wrong, in opposite directions. Two errors caused it:

- *Steps are not independent trials.* Consecutive steps of one generation
  share a prefix, so a single trajectory's N steps carry far less information
  than N Bernoulli trials, and any binomial interval over them is too narrow.
- *Failure to reject is not evidence of absence.* The n=256 design had no
  power to see a 5pp effect; calling it falsified overstated the result as
  badly as the n=48 overclaim did.

Independent prompts are the replicates, and pairing removes prompt
difficulty — which dominates the spread. That design finds the effect at
t=6.34 where the previous one could not see it at all. Repeats do NOT help:
greedy decoding is deterministic, so re-running a prompt reproduces its
acceptance to four decimals and adds nothing.

### Mixed-bit heads: 41% smaller for nothing

The 512-expert MoE stack is 4.688 of the head's 4.856 GiB — **96.5%, in two
tensors** — so head size is essentially one dial, and protecting the other
3.5% at a high bit-width is nearly free (+0.02 GiB from 6- to 8-bit across all
of it). `vqlab mtp-pack --expert-bits` sets the experts independently; the
recipe is recorded in the sidecar and replayed on load.

Paired across the same 12 prompts, `align="committed"`:

| head | resident | acceptance | vs q6 | t |
|------|----------|-----------|-------|---|
| q6 (uniform 6-bit)   | 2.12 GiB | 0.8171 | — | — |
| experts q4 / rest q8 | 1.55 GiB | 0.8105 | −0.65pp | 2.06 |
| experts q3 / rest q8 | **1.25 GiB** | 0.8151 | −0.20pp | 0.54 |

None of the differences is significant at 11 df, and `e3q8` scores *above*
`e4q8` — which a strictly coarser quantization cannot genuinely do, and is
the same tell the ledger used when 6-bit appeared to beat bf16. **Experts at
3-bit costs 41% of the head's residency and buys no measurable acceptance
loss**, which matters most on the large rungs where headroom decides whether
the head ships at all.

The search is safe by construction: head precision **cannot** affect output
quality, because the trunk verifies every drafted token. A coarser head costs
a rejection, never a wrong token. There is no quality gate to defend here,
only acceptance — and acceptance is measured directly.

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

Per-family findings, falsified predictions and open questions live in
[docs/MTP.md](docs/MTP.md).

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
