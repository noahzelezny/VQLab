# Review brief — VQLab, pre-push

For the reviewing sessions. This repo is a **copy** of the quantlab pipeline
prepared for public release. It has never been pushed anywhere. Nothing here
is authorized to publish, and nothing here changes quantlab: the assembly was
read-only toward it.

## Run this first

```bash
cd <repo> && pip install -e . && vqlab selftest
```

Expected: **22 passed, 0 failed, 2 skipped** in well under a minute. It
synthesizes a tiny checkpoint and runs the shipped fitter, gate, packer,
manifest, bundle gate and Metal kernels as subprocesses. Every gate is
exercised in both directions. The 2 skips (end-to-end generation, scoring)
need a real multi-GB checkpoint and are reported rather than omitted.

## What is already verified, so you needn't re-derive it

- All 19 CLI subcommands' surfaces load (`vqlab <cmd> --help`).
- Packing is bit-exact: decode(packed) == decode(unpacked), max delta 0.0.
- Seeded fits reproduce; `--seed -1` diverges (both directions measured).
- Outlier gate fails a collapsed tensor and passes a healthy artifact.
- Manifest check fails altered bytes, passes untouched ones.
- Dense bundles carrying both runtimes SERVE on a **simulated stock**
  mlx-lm; the old single-runtime bundle raises ModuleNotFoundError there.
  Bundled and installed kernels agree bit-for-bit.
- No personal paths, hostnames, box names or credentials in shipped code.

## Where review would genuinely help

1. **Claim accuracy.** Every number in README/REPRODUCING was taken from
   `paper/LEDGER.md` only, never EXPERIMENTS.md. Please check them against
   the *current* ledger — it moves daily, and the in-repo `paper/PAPER.md`
   is a snapshot that must be re-pulled immediately before any push.
2. **Whether the fences are stated strongly enough.** The measured limits
   (VQ wins 2.0–4.5 bpw; crossover bracketed 4.5–6.0 on the dense 27B;
   8-bit lossless; prefill ~0.5x affine at 35B) are in the README, but a
   reader skimming for headline results could miss them.
3. **The seeding divergence.** VQLab's MoE fitter seeds by default (Noah's
   call). Upstream quantlab's stays unseeded so E121/E129/E136 remain
   valid. Confirm that split is what you want before publishing, since it
   means the packaged tool is not byte-for-byte the tool that produced the
   paper's MoE artifacts (`--seed -1` restores the historical path exactly).
4. **Gemma.** Fitting code ships; no gemma quality claim appears anywhere.
   Please confirm nothing implies one.
5. **Anything that reads as overclaiming** in METHODOLOGY. Two descriptions
   in this project's own documents were wrong this weekend ("content hash"
   for a 1 MiB prefix hash; "seeded" fitters that were unseeded), both
   because they read as obviously true. Same skepticism here is welcome.

## Known debt, deliberately not fixed

- Lab-diary tone in copied script docstrings (E-numbers, dated incidents).
  Accurate, just informal for a public repo.
- Scripts parse argparse at module scope; the CLI dispatches via `runpy` to
  preserve exact behavior rather than refactoring files that produced
  published numbers.
- `preflight_ram` is macOS-only (`sysctl hw.memsize`) — acceptable given MLX.
- Any dense artifact already on disk carries the OLD single-runtime bundle
  and needs `build-dense` re-run before it would serve on a stock mlx-lm.

## Open decision for Noah

Which artifacts/model cards the README should reference beyond the
`TheDrainFlorist` org link, and whether the paper ships as a snapshot or an
external link.
