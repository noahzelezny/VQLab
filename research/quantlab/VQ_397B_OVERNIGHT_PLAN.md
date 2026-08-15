# E35 M0b — 397B VQ quality proxy: overnight build plan (2026-08-14)

> **STATUS 2026-08-15 — HISTORICAL, EXECUTED.** This night ran; its
> proxies A/B/C/D are measured and recorded in `EXPERIMENTS.md` (E35 M0b).
> Its "NOT tonight" list is now obsolete: the Metal kernel (M1) is BUILT
> and shipped, and the d8/K4096 sweep it deferred ran as E36.
> **Two corrections this plan predates, both important:**
> 1. Its GiB figures are ANALYTIC (`log2(K)/d + 16/group`). Codes are
>    stored in whole bytes, so unpacked reality is bigger — A is 196 GiB
>    not 134, E is 196 not 145. Only K<=256 at d=4 packs exactly. See the
>    stored-vs-analytic table in `EXPERIMENTS.md` section M2.
> 2. Proxies are superseded by real codes artifacts (`vq_397b_codes.py`),
>    which are smaller, faster to fit, and actually runnable. Proxies only
>    existed because no kernel did.
> Current truth: `EXPERIMENTS.md` + `M1_KERNEL_PLAN.md`.

Goal: kill the E33 DILUTION RISK before any kernel work. Measure what
VQ d4/K1024 on the 397B's 2-bit expert region actually buys vs shipped
`struct6-tail30` (wikitext 2.3982 / code 2.5928). Ship bar: wikitext
≤ 2.3614 (spicyneuron 3.5bit @ 165.6 GiB). 35B says -11.4%; E33's gs32
precedent says expect heavy dilution (its -1.7% became -0.33%). Even
-2% clears the bar — but it must be MEASURED, not projected.

## Architecture of the run (proxy, no kernel — same trick as M0)

NOT a full bf16 copy (751G) and NOT a fresh convert. Reuse the M0 swap
trick at 397B scale:

1. **Fit** (`vq_fit_397b.py`, new): for layers 0-29 only, read the bf16
   source's `experts.gate_up_proj` / `experts.down_proj` (~390 GB), fit
   per-tensor d=4 K=1024 codebooks (pure weight space, per-(row,64) fp16
   scale — identical recipe to the 35B win), and write the DECODED bf16
   tensors + per-tensor relerr to a checkpoint dir. Atomic writes
   (tmp+rename), one .npz-free plain safetensors per layer, resumable.
2. **Assemble** (`assemble_vq_397b.py`, new): copy shipped
   `struct6-tail30` shard-by-shard; for layers 0-29 replace each expert's
   {weight,scales,biases} triple with the single decoded bf16 weight, and
   DROP those modules from config.json's `quantization` dict (that is how
   mlx marks a module unquantized — the M0 `--expert-bits 0` artifacts
   prove the loader path). Tail 30-59, structure, routers, vision: byte-
   identical to the shipped artifact. Expected size ~445 GB (2.9 TB free).
3. **Score**: single-box referee on the M3 (instrument of record), both
   corpora, x2 bit-identical. NOTHING else on that GPU while scoring.

Numbers this yields by morning:
- 397B VQ dilution factor (the only number that decides M1)
- mean 397B reconstruction relerr vs the 35B's 0.222 (sanity: same regime)

## M3 vs M4 — decide by measurement, not vibes (P1)

Noah's lean: M4 (idle, cooler, faster single-thread). The catch: **VQ is
not single-thread work** — k-means is pure GPU GEMM, and M3 Ultra's GPU is
~2x the M4 Max's. The M4's real edge would be NOT competing with anything
else. The pinch candidates are (a) GPU GEMM rate, (b) NFS read of ~390 GB
source + write of ~445 GB artifact from the M4 vs local SSD on the M3.

P1 (10 min, run on BOTH): time (i) a [4096x2048]x[2048x4096] fp32 GEMM
loop, (ii) sequential read of one 8.5 GB source shard, (iii) write+fsync
of 4 GB to the artifact destination. Decision rule: projected end-to-end
hours = fit_GEMM_time x layer_count + IO_time; pick the smaller. If they
are within ~25%, take the M4 (idle + thermals, per Noah).

M4 prereqs if chosen: rsync quantlab repo (git clone from the M3 copy),
python venv with mlx/numpy (no optiq needed for VQ), verify the NFS mount
sees "Exo Models", confirm 128 GB box has >30 GB free during fit (largest
tensor 8.6 GB bf16 -> ~17 GB fp32 working set).

## Probes BEFORE the overnight commit (P2-P4, ~30 min total)

- **P2 — swap machinery**: assemble a variant with ONLY layer 0 swapped,
  load via mlx_lm, score wikitext once. Must land within noise of
  tail30's 2.3982 (one layer of 512 experts moves it slightly or nil). This
  proves mixed quantized/bf16 loading + config surgery + index rewrite at
  397B BEFORE 8 unattended hours depend on it.
- **P3 — one-tensor relerr**: fit layer 0 gate_up on the 397B; expect
  ~0.22-0.32 (the 35B regime). Wildly off -> stop and look.
- **P4 — disk + arithmetic dry-run**: predicted artifact size from shapes
  vs `df`; refuse to launch if headroom < 600 GB.

## Overnight sequence (single nohup chain, each step gated on the last)

    fit (30 layers, ~2.5-4 h on M3 GPU; more if M4+NFS) ->
    assemble (~1 h, IO-bound) ->
    referee x2 both corpora on M3 (~10 min) ->
    append results to logs + EXPERIMENTS.md stub

Crash-safety: per-layer checkpoints, atomic renames, resumable fit;
assemble is idempotent (rewrites output dir); scoring runs only after
assemble exits 0. Log everything to logs_vq397b_*.log.

## Morning decision tree

- wikitext ≤ 2.3614: **class win measured** -> M1 kernel work begins, and
  the artifact math (~148 GB real VQ vs 445 GB proxy) ships the debut.
- 2.3614 < wiki < 2.3982: VQ helps but dilution ate the bar -> extend VQ
  to the 3-bit tail (K=4096 = 3.0 bpw, steep-curve evidence says this is
  where the next win is) before touching kernels.
- ≥ 2.3982 (no help): dilution verdict = E33's; VQ joins the falsified
  pile for the 397B and the honest-efficiency publish proceeds. (35B win
  still stands — worth a small-model release on its own.)

## Not tonight

- No Metal kernel (M1) — fresh-head work, gated on tonight's number.
- No K=4096 / d=8 sweeps — only if the decision tree sends us there.
- No exo/loader changes — proxy only.
