# AGENTS.md — vqlab

Instructions for AI coding agents (Claude Code, Codex, Cursor, Cline, Aider,
any other) working in this repo. Agent-agnostic by design.

**This repo is a TOOLKIT, not a pile of experiments.** Nearly every question
you are about to answer already has an instrument here. Building a one-off
script in a scratch directory when a CLI command exists is the single most
common failure mode in this repo's history, and it costs hours and produces
results that nobody can reproduce.

## Do these three things before writing any code

```bash
# If vqlab is not installed in the interpreter that loads your model,
# run the CLI from the repo root with PYTHONPATH=src.
sed -n 1,40p CONTEXT.md             # question -> stage -> command; each src/vqlab/<stage>/CONTEXT.md is that stage's contract
python -m vqlab.cli --help          # 41 commands. Read the list. Twice.
sed -n 1,60p docs/INDEX.md          # what every doc is FOR + whether it still holds
sed -n 1,40p docs/ONBOARDING.md     # the mechanical pass before fitting ANY new family
```

Then grep for the thing you were about to build:

```bash
grep -rli "<the concept>" src/vqlab/ docs/
```

## The instruments that get rebuilt by accident

| question | use this | NOT |
|---|---|---|
| which layers deserve more bits? | `vqlab layer-leverage` — **rank by the JUMP in `traj_rel`, NOT by `local_rel`** (F95: local_rel is isolation damage and is anti-signal; jump-ranked beat it by 0.7-1.0 pt on every corpus) | hand-rolled band ablations |
| compare two rungs that were ALREADY scored, without re-scoring? | `vqlab kl-pair` over their saved `--per-pos-dir` arrays -- symlink both into one dir under distinct names. Any two rungs scored on a SHARED cache saw the same positions and the same teacher, so they pair after the fact at ZERO GPU cost. F141 got a clean K256-vs-K512 twin comparison out of F139's leftovers this way; the lab has been re-scoring pairs it already had | booking a GPU night to re-score a rung whose per-position array is on disk |
| compare an arm that is an ENV VAR, not a directory? | `vqlab kl-pair` — kl-ladder pairs rungs WITHIN one invocation against its first `--rung`, so an env-var arm (VQ_DENSE_SS, VQ_D4_WALK, the DEVX twins) needs two invocations sharing one `--per-pos-dir`, paired after the fact | reading two overlapping SEMs as "no difference" — that is not the paired test and is far more conservative |
| is a numerics change a COST or a GAIN? which path is right? | `vqlab kernel-truth` — an EXACT float64 reconstruction of a VQLinear is a reference no teacher can beat and no teacher is needed for. A ULP or logit DIVERGENCE says two roundings disagree, never which is wrong (F138 shipped a switch off for weeks over this; F146 found the gate scores the WORSE path by 1.43x). Give the reference the same bf16-rounded inputs the kernels get, or you measure a shared error and call it agreement | reading a divergence figure as an error, or parking the question because the bf16 teacher is gone |
| is this thing bandwidth-bound? how many bytes does a token cost? | `vqlab active-bytes` — bills every tensor by how a DECODE STEP reads it (dense / routed top-k / gathered rows). **Never quote an effective-bandwidth number without it**: F22 counted the expert stack alone, understated Flash's traffic ~9x, and its "large fixed cost" was the missing denominator (F130) | counting the quantized tensors and calling it the model |
| where does the decode time GO, stage by stage? | `vqlab decode-timeline` — the plain measurement the lab never had: every stage of one decode token, in order, timed, **summing to the whole**. Deletion arms give upper bounds that do NOT sum; you cannot rank what you have not partitioned. It measures CUMULATIVE PREFIXES (one real eval each), never per-op `mx.eval` — that is F133, which reported parts summing to 1667 us against a 410 us whole. It REFUSES to rank unless the deltas add up AND no delta is negative | timing each module with its own `mx.eval` and adding them up |
| which COMPONENT owns the decode time? | `vqlab decode-ladder` — per-component deletion arms with an output checksum beside every timing | a fresh deletion script that seeds `cache.keys` and crashes on Flash's ArraysCache/QSAKVCache |
| how much damage does this artifact carry? | `vqlab score` / `kl_damage.py` | ad-hoc KL scripts |
| ppl on the house corpora | `scripts/score_ppl_resident.py` + the THREE corpora in `src/vqlab/score/referee/` (prose / code-public / literary) | your own corpus files |
| task benchmarks | `vqlab tasks` (src/vqlab/score/score_tasks_streaming.py; layer-streamed, scores models larger than RAM) | a new eval harness |
| is this artifact releasable? | `vqlab check-release` / `check-bundle` / `selftest` | eyeballing |
| fit a mixed-geometry rung | `vqlab fit-moe --vq-layers` (scatter fits) | bespoke build scripts |
| rebuild an artifact at a new per-layer geometry | `vqlab geo-build` (diff-style: named modules refit from the bf16 teacher, everything else keeps shipped bytes; REFUSES ragged nsub, sets pack_bits, verifies fit reuse by codebook shape) | hand-rolled build scripts in scratch |
| how many layers to promote / demote? | `vqlab alloc-sweep` — measures the COST and VALUE curves one variable at a time and prints the marginal-ppl-per-100MB frontier | picking a number, building it, and generalizing from n=1 |

If an instrument genuinely does not exist, **add it to `src/vqlab/` with a CLI
entry** rather than leaving a script in scratch. That is why the toolkit is
good: every past agent who needed something added it here.

## Agents: use the MCP, not the shell

**Reserve your F-number, do not just read it.** `next_f_number` used to be
`max(log)+1`, a pure read; the log only changes at COMMIT time, so two
sessions running experiments in parallel both got 151 on 2026-09-19 and both
wrote an entry. Call it with `reserve=true` before a long experiment -- it
claims the number atomically (O_EXCL lock, 6 h TTL, released by
`findings_append`, or by hand with `release_f_number`). A bare read still
works and still burns nothing, and now reports which numbers are held.

`vqlab mcp` serves this box's lab over MCP (stdio JSON-RPC, stdlib only; one
server per box, like exo). Tools: `where_is` (deterministic lookup over the
configured storage roots, `vqlab.config` — use it before ever claiming
something is missing),
`list_artifacts`, `artifact_config`, `read_doc`, `run` (allowlisted
subcommands, detached, under the GPU lease, refuses paths outside the
configured storage and refuses while an exo instance is placed), `status`, `stop`, `list_runs`,
`gpu_state`, `next_f_number`, `findings_tail`, `findings_append` (the only
tool-side writer of the findings log: every field required, prediction
recorded verbatim). `publish` is not exposed; it is a human's action.

    PYTHONPATH=src python -m vqlab.cli mcp --list
    PYTHONPATH=src python -m vqlab.cli mcp --call where_is '{"name":"teacher_topk"}'

## Authority order when sources disagree

1. **The shipped artifact's own `config.json` and `README.md`** — the record
   of what actually shipped. A card's methodology section says how the mix
   was chosen; the config says what the mix IS.
2. The findings log — the measured record (F-numbers, corrections applied
   in place). It is the lab's own record, kept outside the public repo:
   `vqlab mcp` reads and appends `lab/FINDINGS-LOG.md` (or
   `$VQLAB_FINDINGS_LOG`). `docs/FINDINGS.md` carries its conclusions.
3. Anything narrative (plans, notebooks, session notes) — it records
   attempts, including ones later overturned. **Never characterize a
   released artifact from a narrative source** (this error was made twice in
   one hour on 2026-09-13; both times a 30-second config read would have
   prevented it).

## Standing rules that have each been paid for

* **Perplexity is deterministic.** Re-scoring an artifact returns the same
  number; it cannot estimate a noise floor. A fit-to-fit floor requires a
  SECOND INDEPENDENT FIT of the same recipe.
* **Isolation probes are anti-signal for allocation.** Measuring one
  layer's damage against an intact network is the condition where
  downstream laundering hides it (quantlab E12 on GLM/affine; F93-F95 on
  Flash/VQ). Use compounding/trajectory measures.
* **Rank allocation by KL, not ppl.** Ppl aggregates and absorbs offsetting
  errors. KL is the ranking instrument, and since F118 (Noah, 2026-09-16)
  it is also the RELEASE GATE on Flash-Next: `vqlab kl-ladder`, paired,
  three corpora at 12288, |t|>2. Ppl is printed on the card with its sign,
  not gated -- it inverted against KL on every winning arm of F116-F118.
* **One harness.** Never compare a number from one scoring path against
  another. qwen4_exp loglikelihoods even shift 0.1-0.7 nats with batch
  composition (F87) — same path AND same batching.
* **Depth/geometry laws are family-local.** GLM, 397B and Flash each measured
  a different shape. Never inherit an allocation across families.
* **A ULP figure is DISAGREEMENT, not error (F138).** `VQ_DENSE_SS`'s "up to 8.00 ULP" was read as a quality cost for weeks and shipped the switch OFF. Measured, the tree reduction is the BETTER-rounded one — no corpus worse, prose -0.605 mnats at |t|=4.31 — because a tree's error grows O(log n) against a serial chain's O(n). Before treating a ULP divergence as a cost, ask WHICH rounding is closer to the teacher. Nobody had.
* **...NOR ON MoE (F164).** The MoE gate is `VQ_FUSED_MAX_N = 4096` and N
  counts (token, expert) PAIRS, not tokens: Flash-Next at top_k=10 and
  chunk 512 is N=5120, just over. **chunk <= 409 reaches the fused expert
  kernel.** So the blind spot is fleet-wide, and no published number on any
  artifact describes the kernels that run at generation. For THIS family
  the fix is available -- the bf16 teacher is on the HDD, so decode-path
  ABSOLUTE KL is buildable; chunk-384 caches are ~25-30 min. Mind F111:
  the metric is not chunk-invariant, so those numbers are a new harness and
  do not slot into existing card tables.
  **Scope this claim carefully.** It is about the KL/ppl RELEASE GATE only.
  `smoke`, `decode-ladder` and serving all run at N=1 and DO exercise the
  fused kernels; F144 scored quality at chunk 8 inside the gates. Saying
  "nothing has scored the shipped kernels" is wrong and was written into
  F164 before being corrected.
* **The KL gate does NOT exercise the DECODE kernels on dense artifacts
  (F137).** `kl-ladder` scores at the cache's chunk (512, correctly -- F111);
  the fused decode path is gated at `N <= 32` for packed d4, so scoring falls
  through to `_decode_matmul` (wdec + GEMM). Every decode-path numerics change
  -- `VQ_DENSE_SS` (up to 8 ULP), `VQ_D4_WALK`, the DEVX twins -- passes the
  referee UNTESTED, because the referee never runs that code. One artifact
  ships two numerically distinct paths (measured 0.87 mnats apart on prose)
  and the gate scores one. Diagnose with `VQ_DENSE_FUSED_MAX_N=1024`, which
  forces the fused path at scoring N -- but that is a DIAGNOSTIC, not the
  shipped config. NOT checked for MoE artifacts; do not assume the fleet is
  covered.
* **An artifact can change under you.** Rule III protects against a busy
  GPU; it does not protect against ANOTHER SESSION rebundling the artifact
  you are measuring. F151 lost 3 of 13 runs that way, and no power gate can
  see a filesystem write. Before a long campaign, record
  `stat -f %m <artifact>/model.py` and re-check it at the end; timestamp
  every run so the boundary is recoverable if it happens anyway.
* **Generate one token through the shipping runtime** before calling anything
  releasable (rule III.11 — an unservable artifact once scored perfectly).
* **Artifacts go to configured storage, never the system disk or the repo.**
  `vqlab.config` resolves scratch, models, teachers and the fit store (env
  var, then `~/.config/vqlab/config.toml`); teachers are archived there, not
  re-downloaded.
* **Version vocabulary:** v1 = first codebooks; v2 = mixed codebooks
  (measured per-layer allocation); v3 reserved for gradient-tuned. Artifacts
  never carry campaign letters.

## The law book — `docs/FINDINGS.md`

**Read it before proposing any quantization idea.** Five sections:
I settled laws, II retracted leads (do NOT re-chase), III instrument rules,
IV MLX/Metal rules, V open questions. Each law survived at least one attempt
to kill it, and each Metal rule cost at least one run. The most load-bearing,
distilled — the file itself is authoritative:

**Quality / allocation laws**
* **Position law (I.2)** — early layers tolerate cheap bits; enrichment pays
  only in the BACK of the network; knee ≈ layer 30 of 60 (affine) / L10 (VQ
  shallow-harvest). It TRANSFERS across quantization families. The file says
  "do not rediscover it a third time"; it was rediscovered a fourth time on
  2026-09-13 (F83/F93) by probing bands by hand. Read the law first.
* **Escape the cheapest width broadly before enriching narrowly (I.3)** —
  under a byte budget, maximize non-cheapest layers; never buy the expensive
  width while any layer sits at the floor.
* **Fit error != output damage (I.6)** — weight-space relerr does not rank
  output quality across geometries. Only KL/ppl on the ASSEMBLED model counts.
  (Re-confirmed the hard way in F78/F94: every proxy inverted.)
* **Higher d wins at MATCHED rate (I.10)** — d4 > d2 by ~12% KL at 2.00 bpw,
  confirmed on a second exact rate twin at 3.00 bpw. Modest, not a landslide,
  and only meaningful at MATCHED rate.
* **Quality tracks total bytes (I.1)** — packaging washes across K at d4;
  whether it washes across d is UNSETTLED.
* **Price a rung before fitting it (I.5)** — the size model predicts, and
  every data point must be stamped pre- or post-vision-graft (the tower is a
  fixed 0.849 GiB; mixing the two is a units mismatch that once looked like
  a geometry effect).
* **Operational sweet spot d4/K256 (I.8)** — and **healthy relerr ranges
  SCALE WITH K** (K2048 ~0.19, K256 ~0.31, K128 ~0.46). Set
  `--relerr-abort` PER GEOMETRY; a threshold tuned at one K wrongly aborts
  healthy fits at another.
* **Decode is a wash across geometries; PREFILL is where geometry shows
  (I.9)** — non-byte-aligned code widths pay bit-extraction.

**Instrument rules (III) — violations produced every false result in that file**
* Pre-register predictions before fitting or scoring; a falsified prediction
  is recorded as falsified, never reframed.
* **A comparison row must name the ARTIFACT and the INSTRUMENT that produced
  it.** A number older than the artifact it faces gets RE-MEASURED, not
  cited. Never compare a real artifact against a proxy score.
* Comparators must pass `check_comparator.py` before their row is believed —
  a comparator that loads short scores worse and FLATTERS us.
* Speed: n>=3 with scatter, prompt length stated, one process per arm, never
  on a contended box. **Quote a RATIO between arms from the same session,
  never an absolute** — at ~100 GiB the decode instrument is BIMODAL
  (21.1 / 12.7 / 21.3 / 21.2 tok/s, same artifact, back to back; cause
  unknown).

**Metal rules (IV)**
* **Fused d4/d2 kernels cache the codebook in threadgroup memory:
  `K * dim * 2 < 32768` is a HARD architectural ceiling** (d4 safe to K2048,
  d2 to K4096; K4096@d4 fails ON the cap). Compute `K*dim*2` FIRST.
  `XPC_ERROR_CONNECTION_INTERRUPTED` is how Metal reports this
  over-allocation — it is NOT a compiler-service fault.
* **Dense and MoE are DIFFERENT RUNTIMES** — `vq_dense.py`'s fused path is
  gated on `codebook.shape[1] == 2`. A smoke on one path says NOTHING about
  the other.
* Load under `with mx.stream(mx.cpu):` with `mx.eval` INSIDE the block — a
  lazy read still pending when a save forces evaluation is paid inside a GPU
  command buffer and gets watchdog-killed "at the write step". **Binding is
  set when the read op is CREATED, so this must happen at LOAD; wrapping a
  later `mx.eval` does not rebind it** (F120: 12.2 GiB blocks on the 397B
  teacher died in `eval_params_budgeted` one layer further each run as the
  page cache warmed). **And the CPU-stream load is NOT arithmetic-neutral** —
  it moved Flash-3.2 prose from 156.7034 to 155.1233 KL — so it is opt-in per
  family (`cpu_stream_load`) and every published qwen4_exp number keeps the
  GPU-stream path. **A performance or memory fix is an instrument change
  until an A/B says otherwise**; the A/B costs one cell, twice.

**Retracted (II) — do not re-chase without new evidence:** "cheap-shallow
beats the rung above it" (proxy-score artifact), "VQ beats 8-bit affine on
embeddings" (confounded by an fp32 path), the fused row-gather prefill lever
(MLX already fuses it), byte-aligned packing (0 bytes saved, 37% decode cost).

## Before scoping ANY engineering change: audit the fleet

One query over every shipped `config.json` costs 20 seconds and routinely
shows the problem already solved elsewhere. F96 concluded "this needs a
kernel change" by reasoning about a format in isolation; F97's fleet audit
found 18 of 19 artifacts packing exactly and reduced it to one rung's
geometry choice — a refit, not kernel work. Corollary: if one artifact is
the only one with a problem, suspect that artifact's config, not the
shared machinery.

## Two agents, one artifact root

Storage roots are SHARED. More than one session works this repo at a
time, and a bundle rewrite is a write to another session's experiment.

**Measuring? PIN, then SMOKE, then measure.** Do not benchmark or score the
live artifact directory. Build a pinned copy -- a dir of symlinks to the
safetensors plus the ONE `model.py` you mean to measure -- and two symlink
trees later a concurrent rebundle is structurally unable to enter your
experiment. The smoke step is not optional politeness: pinning freezes
whatever you pinned, INCLUDING a bundle that was already broken when you
copied it. That is not hypothetical -- the session that invented this
technique pinned a twin whose rebundle had pulled in mid-flight arch code, and
caught it only because `vqlab smoke` ran on the copy first. A pin without a
smoke buys you a reproducible wrong number. On 2026-09-19 a fleet-wide rebundle landed mid-run and voided 3 of
13 arms of a decode campaign (F151); the same session finished the rest of that
campaign from a pinned copy while the repair pass was still running.

**Rewriting artifacts? Say so first, and gate ONE per family before the
second.** Announce a fleet-wide write before starting it. Then run BOTH
`vqlab smoke --max-tokens 8` (text, ~30 s) and `vqlab vision-smoke` on one
artifact of each family before touching the next. F154 is what skipping this
costs: a vision repair was verified on the family it was written for, applied
to 18 artifacts, and silently broke text generation on 3 of them and loading
outright on 4 more. A new gate does not excuse re-running the old one -- the
gate you just wrote is aimed at the failure you already found.

**An absence observed through one access path is not an absence.** Three
times on 2026-09-19 an instrument's REACH was reported as a property of the
artifact: an exact-zero KL read as "no difference" (an unexercised code path,
F143); a conv3d failure read as "transposed tower" (the test had skipped the
loader's sanitize(), F161); an AttributeError read as "module not defined"
(different spelling and container type in the other runtime, F163). Before
declaring something absent, wrong or unchanged, confirm the instrument
actually traversed the path the runtime traverses -- read the arch's
`__init__` for the module, call the loader's own entry point, force the
code path open. A gate that never ran the code reports the gate.

**Releasing? The baseline is the HF revision, not the local copy.**
Local serving copies drift: 8 of 20 differed from
published on 2026-09-19, mostly local v2 against published v1.5. Pull the real
one first -- `hf download <repo> model.py --local-dir /tmp/hfcheck/<repo>` costs
nothing -- and rebundle from it, or you publish a runtime nobody scored.

## Long runs

**Use `vqlab queue run <file> --preflight`, then `--detach`** (src/vqlab/agents/run_queue.py). It pins the code in a worktree (no tree freeze), holds the GPU lease, retries resumable builds, refuses unsmoked pins (`vqlab pin`), and fails loudly. Hand-rolled chains are what night 4 (2026-09-26) paid for. The text below is why it exists.

Overnight/multi-hour work needs `nohup ... & disown` plus per-module
checkpoints — a chain tied to the session dies with it. Geometry refits crash
on GPU timeouts under disk contention; wrap them in a retry supervisor that
resumes from checkpoints.
