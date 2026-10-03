# CONTEXT — where do I go?

Routing for humans and agents. `AGENTS.md` covers who we are and the rules;
this file says **which instrument answers your question**. Each stage below
is one job. Read that stage's section, run the command it names, and don't
load the rest of the repo. If your question isn't in this table, grep
`python -m vqlab.cli --help` before you write anything new: rebuilding an
instrument that already exists is this repo's #1 failure mode.

Run commands as `python -m vqlab.cli <cmd>`, or with `PYTHONPATH=src` from the
repo root when vqlab is not installed in the active interpreter. `<cmd> --help` shows each command's full surface.

## Pipeline stages

Stages run in the order listed. Every stage reads an artifact directory or a
teacher, writes a new directory (never in place), and leaves a record.

| # | stage | question it answers | commands |
|---|---|---|---|
| 1 | **plan** | What will it cost? Where should the bits go? What IS this model? | `onboard` (sequence a new teacher: profile -> caches -> determinism -> init sweep), `family-profile` (headers only: legal (d,K), GiB per bit, module signatures), `price`, `zero-groups` (code bytes spent on the teacher's near-zero rows; 397B ~10%), `layer-leverage` (rank by the traj_rel JUMP), `loo-bands` (restore one band to exact per hybrid, score in the assembled network: the damage map), `alloc-sweep`, `probe-init`, `preflight-ram`, `preflight-disk`, `mtp-probe` / `mtp-probe35` |
| 2 | **fit** | teacher weights -> codebooks + codes | `fits` (find / reuse stored fits FIRST), `fit-moe`, `fit-dense`, `fit-ple`, `geo-build` (refit named modules, keep the rest), `fit-additive` (two small codebooks expanded to one ordinary K1*K2 fit: a KL test with no new kernel), `reselect` (re-pick codes under a self-generated activation Gram, codebook fixed; F67-F78, re-test under the KL gate), `harvest-parts` |
| 3 | **assemble** | codes -> an artifact that loads | `pack`, `pack-dense`, `pack-ple`, `splice-ple`, `ple-swap`, `unpack-dense`, `reskeleton`, `minibase` (a band's shards as a fit base) and `mix` (per-layer-band sources -> one artifact, shard by shard), `stream-convert`, `teacher-prep` (an official release -> an exact teacher in one command: relabel + `sanitize-stream`, one layer at a time), `build-dense`, `graft`, `graft-extras` (DeepSeek Vision-Exp tower + image routing from the release), `mtp-extract` / `mtp-pack` / `mtp-graft`, `mtp-head-ds4` (DeepSeek-V4-Flash's head), `mtp-arms` (pinned per-head arms for `speed-pair-knurlogic --draft`) |
| 4 | **bundle** | Ship the runtime inside the artifact | `bundle` (MoE), `rebundle-dense`, `patch-arch`, `vision-layout` |
| 5 | **gate** | Is it loadable, correct, and releasable? | `check`, `check-release`, `check-bundle`, `bundle-accept`, `verify`, `smoke`, `vision-smoke`, `check-comparator`, `selftest`, `validate` (overnight queue), `pin` (freeze + smoke a copy before measuring), `mtp-smoke-head`, `spelling` (US spelling in released text) |
| 6 | **score** | How much damage does it carry? | `kl-ladder` (the release gate), `kl-pair` (pair two runs after the fact, zero GPU), `kl`, `score` (ppl, printed, not gated), `stream-score` (the layer-streamed scorer kl-ladder runs; also builds teacher caches), `tasks` (task benchmarks), `kernel-truth`, `kernel-truth-moe` |
| 7 | **bench** | How fast is it, and where does the time go? | `decode-timeline`, `prefill-timeline`, `decode-ladder`, `active-bytes`, `prefill-bench`, `coverage`, `host-attrib`, `hc-micro`, `mtp-bench`, `mtp-accept`, `speed-pair` (two arms, fresh process each, ratio), `speed-pair-knurlogic` (same, served by Knurlogic: one Mac or a pipeline split) |
| 8 | **ship / serve** | Publish or serve it | `publish` (a human action), `serve`, `mtp-generate` |

Skip-zero, `src/vqlab/skipzero/` (a shipped format since 2026-09-29, served natively by the runtime): `sz-pack` / `sz-check` / `sz-resident` / `sz-bitexact` (vq-skipzero: fully-dead VQ rows dropped on disk; 397B -10.8% disk / -11.1% resident at identical KL; docs/SKIPZERO.md).

Around the pipeline: `release-prep` (sizes, junk, provenance, gate, Hub diff, then the exact `publish` line), `size` (text / +tower / +MTP, the card's three numbers), `card-tables` (the card's KL table from the scorer's JSON), `config` (where vqlab reads and writes; `config init` on a new machine), `queue` (run a list of steps from pinned code under the GPU lease; `--preflight` first; `status` shows an ETA from the data; `--on BOX` checks paths read on that box first), `reserve` (who has a box until when; `queue run` refuses a box reserved for someone else), `mcp` (the lab over MCP for agents), `gui` (read-only local window).

## Records: nothing happens without one

| record | where | what | read with |
|---|---|---|---|
| **run log** | `~/.vqlab/runs.jsonl` (per user, every `vqlab` call) | argv, commit, dirty files, library versions, exit code, duration | `vqlab runs` |
| **score stamp** | the `measured` block in every scorer's output | which artifact a number is OF: build-record id, pin, runtime md5/mtime, byte fingerprint, run id | read the score's JSON |
| **build record** | `<artifact>/vqlab_provenance.json` | how these bytes were made: fitter settings, inputs + lineage, per-module origin, runtime profile, hashes | `vqlab provenance <artifact>` |
| **tamper stamp** | `manifests/` (outside the artifact) | were the shard bytes rewritten? (for artifacts built before build records existed) | `vqlab manifest check` |
| **fit store** | `<store>/fits/<family>/<teacher>/L<layer>/<proj>/d<D>-K<K>/<fit_id>` + `index.jsonl` (HDD archive; `$VQLAB_FIT_STORE`) | every fitted module with its recipe; fitters file into it, `geo-build --pool` reuses from it (recipe must match) | `vqlab fits list / stats / census / retag` |
| **artifact registry** | `registry/artifacts.jsonl`, `registry/hub.jsonl` (in git) | every artifact from provable facts (runtime, geometry, size, fingerprint, record id); latest Hub comparison | `vqlab registry list / hub` |
| **queues** | `~/.vqlab/queues/<stamp>-<name>/` | each step's cmd, stdout, stderr, verdict; the pinned commit | `vqlab queue status / list` |
| **box reservations** | `<scratch>/vqlab-reservations.json` (the shared SSD; `$VQLAB_RESERVATIONS`; local fallback `~/.vqlab/reservations.json`) | who has each box until when, and why | `vqlab reserve --list`, MCP `gpu_state` |
| **family data** | `families/<family>/` | `entry.json` (how to read the tensors, no code change) and `teachers/<teacher>/{profile,onboard,teacher_caches}.json` | `vqlab family-profile`, `vqlab onboard`, MCP `where_is` |
| **findings** | `lab/FINDINGS-LOG.md` (private; `$VQLAB_FINDINGS_LOG`) | every measured result, F-numbered | MCP `findings_tail`; reserve a number with `next_f_number reserve=true` |

Design: `docs/PROVENANCE.md`.

## Reference (stable across runs: read it, treat it as constraints)

- `AGENTS.md`: the lab's rules and the authority order when sources disagree
- `docs/FINDINGS.md`: the law book (settled laws, retracted leads, instrument rules, Metal rules)
- `docs/INDEX.md`: what every doc is for, and whether it still holds
- `docs/ONBOARDING.md`: the mechanical pass to run before fitting a new family
- `METHODOLOGY.md`: read it before publishing any number

## The paper

- `research/CONTEXT.md`: the paper's source and the scripts that regenerate its tables.

## Code layout (`src/vqlab/`)

One folder per stage, each with a `CONTEXT.md` contract (Inputs / Process /
Outputs / Rules that bite). Read the one for your stage and skip the rest.

```
src/vqlab/
  cli.py        every command -> its script (python -m vqlab.cli <cmd>)
  _layout.py    stage folders on sys.path, runtime_file()/find(), pre-split import aliases
  runtime/      SHIPPED code, spliced verbatim into every model.py
  core/         shared libraries (families registry, source loaders)
  plan/  fit/  assemble/  bundle/  gate/  score/  bench/    the pipeline, in order
  records/      run log + build records + score stamps + artifact registry + tamper stamp
  ship/         publish (a human action) + serve
  agents/       MCP server, queue runner, read-only GUI
  mtp/          MTP speculative-decoding LIBRARY (vqlab.mtp); its tools live in the stages
  family/       model families as plugins, one file each (fit layout, streamed scorer, tokenizer hook); deepseek_v4 first
  skipzero/     the vq-skipzero pack/check tools (the runtime switch itself lives in runtime/)
```

Tools are standalone scripts that import siblings by bare name (`import
vq_pack`); `_layout` makes that work across folders. Old dotted names
(`vqlab.vq_switch`, `vqlab.geo_build`, ...) still import, because published
runtime text uses them. **`runtime/` is the one hard boundary**: editing it
changes what `check-bundle` compares every published artifact against.
