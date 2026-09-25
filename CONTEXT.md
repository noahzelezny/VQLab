# CONTEXT — where do I go?

Routing for humans and agents. `AGENTS.md` covers who we are and the rules;
this file says **which instrument answers your question**. Each stage below
is one job. Read that stage's section, run the command it names, and don't
load the rest of the repo. If your question isn't in this table, grep
`python -m vqlab.cli --help` before you write anything new: rebuilding an
instrument that already exists is this repo's #1 failure mode.

Run commands as `python -m vqlab.cli <cmd>`, or with `PYTHONPATH=src` from the
repo root in the exo env. `<cmd> --help` shows each command's full surface.

## Pipeline stages

Stages run in the order listed. Every stage reads an artifact directory or a
teacher, writes a new directory (never in place), and leaves a record.

| # | stage | question it answers | commands |
|---|---|---|---|
| 1 | **plan** | What will it cost? Where should the bits go? | `price`, `layer-leverage` (rank by the traj_rel JUMP), `alloc-sweep`, `probe-init`, `preflight-ram`, `preflight-disk` |
| 2 | **fit** | teacher weights -> codebooks + codes | `fit-moe`, `fit-dense`, `fit-ple`, `geo-build` (refit named modules, keep the rest), `harvest-parts` |
| 3 | **build** | codes -> an artifact that loads | `pack`, `pack-dense`, `pack-ple`, `splice-ple`, `stream-convert`, `build-dense`, `graft`, `mtp-extract` / `mtp-pack` / `mtp-graft` |
| 4 | **bundle** | Ship the runtime inside the artifact | `bundle` (MoE), `rebundle-dense`, `patch-arch`, `vision-layout` |
| 5 | **gate** | Is it loadable, correct, and releasable? | `check`, `check-release`, `check-bundle`, `bundle-accept`, `verify`, `smoke`, `vision-smoke`, `check-comparator`, `selftest`, `validate` (overnight queue) |
| 6 | **score** | How much damage does it carry? | `kl-ladder` (the release gate), `kl-pair` (pair two runs after the fact, zero GPU), `kl`, `score` (ppl, printed, not gated), `kernel-truth` |
| 7 | **bench** | How fast is it, and where does the time go? | `decode-timeline`, `decode-ladder`, `active-bytes`, `prefill-bench`, `coverage`, `host-attrib`, `hc-micro`, `mtp-bench`, `mtp-accept` |
| 8 | **ship / serve** | Publish or serve it | `publish` (a human action), `serve`, `mtp-generate` |

## Records: nothing happens without one

| record | where | what | read with |
|---|---|---|---|
| **run log** | `~/.vqlab/runs.jsonl` (per user, every `vqlab` call) | argv, commit, dirty files, library versions, exit code, duration | `vqlab runs` |
| **build record** | `<artifact>/vqlab_provenance.json` | how these bytes were made: fitter settings, inputs + lineage, per-module origin, runtime profile, hashes | `vqlab provenance <artifact>` |
| **tamper stamp** | `manifests/` (outside the artifact) | were the shard bytes rewritten? (for artifacts built before build records existed) | `vqlab manifest check` |
| **findings** | `docs/FINDINGS-LOG.md` | every measured result, F-numbered | MCP `findings_tail`; reserve a number with `next_f_number reserve=true` |

Design: `docs/PROVENANCE.md`.

## Reference (stable across runs: read it, treat it as constraints)

- `AGENTS.md`: the lab's rules and the authority order when sources disagree
- `research/quantlab/FINDINGS.md`: the law book (settled laws, retracted leads, instrument rules, Metal rules)
- `docs/INDEX.md`: what every doc is for, and whether it still holds
- `docs/ONBOARDING.md`: the mechanical pass to run before fitting a new family
- `METHODOLOGY.md`: read it before publishing any number

## Working record (changes every run: process it as input, don't cite it as law)

- `research/quantlab/EXPERIMENTS.md`: the lab notebook. **Never describe a released artifact from it**; read the artifact's `config.json` first.
- dated docs in `docs/` (`*-2026-MM-DD.md`, `MORNING-REPORT-*`)

## Code layout (`src/vqlab/`)

Flat today; the move into stage subpackages is in progress (see
`docs/PROVENANCE.md` §6 and the commit log). **`vq_switch.py`, `vq_dense.py`
and `vq_pack.py` are the SHIPPED runtime**: they are spliced verbatim into every
artifact's `model.py`, so editing them changes what `check-bundle` compares
every published artifact against.
