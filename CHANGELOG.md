# Changelog

## 0.1.0 — 2026-10-19

First release. VQLab fits size-targeted, data-free vector-quantization
codebooks for large language models on Apple Silicon (MLX), assembles and
bundles the quantized artifacts with their Metal runtime, and scores them
against their bf16 teachers (paired KL on three corpora, perplexity, task
benchmarks) before release. One `vqlab` CLI covers plan, fit, build, bundle,
gate, score, bench and ship; `vqlab gui` is a local page for the same work;
`vqlab mcp` exposes the lab to agents.

Licensed AGPL-3.0-or-later. Model artifacts follow their base models'
licenses.

### Installing and setup
- `pip install vqlab` needs no checkout: the first command creates a lab
  folder and a config, and `vqlab selftest` runs from the installed wheel.
- `vqlab lab config` shows every storage location, where it resolved from and
  its free space; `vqlab lab config set` and the GUI's Storage page change them.
  Nothing assumes a disk layout.
- Dependencies are pinned to the tested ranges (mlx 0.32.x, mlx-lm
  0.31-0.32, knurlogic 0.1.x from 0.1.8).
- `vqlab --version`; `vqlab lab doctor` says when a newer release is out
  (PyPI, asked at most once a day; `VQLAB_OFFLINE=1` skips it).
- `vqlab lab report-bug` fills in a GitHub issue with the versions, chip and
  last failed command (no paths) and prints the link; nothing is sent.

### The GUI (`vqlab gui`)
- Library: published builds and your own, with their measured KL.
- Recipe: pick a model (downloaded for you if it is not on this machine),
  size a build on the published builds' KL curve, and compile it into a
  queue; Qwen3.5/3.6 MoE, Qwen3.8 dense, Qwen3.8-Flash-Next, GLM-5.3-Flash
  and Gemma 4 (26B and E4B) build from the page.
- Jobs: run a queue's preflight, then the build, and follow it.
- An "Update available" note when a newer release is out (click copies the
  upgrade command), and a "Report a bug" link that opens a prefilled issue.

### Families
- Qwen3.5/3.6 MoE (35B, 397B), Qwen3.8 dense, Qwen3.8-Flash-Next,
  GLM-5.3-Flash, Gemma 4, and DeepSeek-V4-Flash (FP4 experts read natively;
  `vqlab teacher-prep` turns the official release into an exact MLX
  teacher).
- Multi-token-prediction draft heads extracted, packed and smoked with the
  build.

### Building artifacts
- `vqlab fit-moe` / `fit-dense` fit codebooks; `vqlab geo-build` refits
  named modules and keeps every other shipped byte; `vqlab mix`,
  `loo-bands`, `minibase` and `reskeleton` build mixed-geometry artifacts.
- vq-skipzero: fully dead VQ rows are dropped on disk and served natively
  (397B: -10.8% disk, -11.1% resident, KL unchanged).
- Every large writer checks free disk first; fits are filed in a fit store
  and reused rather than refit.

### Measuring and release gates
- `vqlab kl-ladder` is the release gate (paired, three corpora, |t|>2);
  `vqlab kl-pair` pairs already-scored builds at zero GPU cost.
- `vqlab decode-timeline`, `prefill-timeline`, `active-bytes`,
  `kernel-truth` and `speed-pair` measure where time and bytes go.
- `vqlab check-release` fails an artifact whose tokenizer does not tokenize
  like its base model's, or whose files name the machine that built it;
  `vqlab publish` refuses the same leaks. Every artifact carries a portable
  build record (`vqlab provenance`).

### Runtime (ships inside each artifact)
- Fused Metal decode and prefill kernels for d2/d4/d8 codebooks
  and multi-token-prediction drafting.

### Running long jobs
- `vqlab queue run` pins the code, holds the GPU lease, runs a small real
  preflight of every step, retries resumable builds and fails loudly;
  `vqlab queue wait` blocks until a queue finishes.
