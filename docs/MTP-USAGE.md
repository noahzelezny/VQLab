# MTP speculative decoding — usage and measurements

Moved verbatim from README.md on 2026-09-15 (it was half the README).
Per-family findings and open questions live in [MTP.md](MTP.md); the
exo spec in [MTP-EXO-SPEC.md](MTP-EXO-SPEC.md).


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
vqlab publish      --artifact <dir> --repo <owner/name>           # gated upload
vqlab mtp-extract  --src <bf16 checkpoint> --out <graft.safetensors>  # pull the head
vqlab mtp-pack     --model <artifact> --mtp <graft.safetensors>       # build sidecar
vqlab mtp-generate --model <artifact> --temp 0.7
vqlab mtp-accept   --model <artifact> --head q8=<sidecar>        # acceptance
vqlab mtp-bench    --model <artifact> --tokens 128               # speedup
```

### Supported families

A family is one table entry in `vqlab/mtp/registry.py` plus a head module;
nothing else in the package names an architecture.

| family | models | head module | measured |
|---|---|---|---|
| `qwen4_exp` | Qwen3.8-Flash-Next | `mtp_head.py` | 0.78–0.82 acceptance, **1.58x**, 1.25 GiB head |
| `qwen3_5` | Qwen3.8-27B (dense) | `mtp_head_qwen35.py` | 0.63–0.85 acceptance, 0.49 GiB head |
| `qwen3_5_moe` | Qwen3.5-397B-A17B, Qwen3.6-35B | `mtp_head_qwen35.py` | 3.19 GiB head (experts 3-bit) |

The qwen3_5 head has three wiring choices the checkpoint does not determine —
the RMSNorm delta convention, the concat order into the fused `fc`, and which
side of the trunk's final norm it reads. Each wrong choice drafts at chance
with no error, so `vqlab mtp-probe35` sweeps them against one model load
instead of trusting an argument. See [docs/MTP.md](docs/MTP.md).

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

### What the speedup depends on

**The MTP speedup is a function of how predictable your text is, not a
property of the model.** The head drafts a token, the trunk verifies it, and
the gain is however often the draft was right. Measured on one model with one
head, acceptance ranged **0.64 to 0.95 across twelve ordinary prompts**, and
hit exactly **1.0** on repetitive synthetic text.

| workload | acceptance | speedup |
|----------|-----------|---------|
| hardest prompt measured | 0.64 | ~1.35x (implied) |
| typical prose questions | 0.78-0.82 | **1.58-1.65x (measured)** |
| easiest prompt measured | 0.95 | ~1.87x (implied) |
| repetitive / boilerplate | 1.00 | 1.95x (measured) |

Only the bolded row and the last are measured directly; the others are
implied by interpolating between them. The practical reading: the feature is
strongest on code completion, structured output and boilerplate — where
drafts land — and weakest on short high-entropy answers, where decode time
matters least anyway.

The same fact makes **cross-project acceptance numbers meaningless without
shared prompts**: the spread from workload alone (0.64-1.0) is wider than the
gaps usually quoted between implementations.

Trunk quantization, by contrast, does NOT affect acceptance — measured across
three rungs, paired, nothing significant. See [docs/MTP.md](docs/MTP.md).

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
