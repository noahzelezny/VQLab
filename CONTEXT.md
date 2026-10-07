# CONTEXT — where do I go?

Routing for humans and agents. `AGENTS.md` covers who we are and the rules;
this file says **which instrument answers your question**. Each stage below
is one job. Read that stage's section, run the command it names, and don't
load the rest of the repo. If your question isn't in this table, grep
`python -m vqlab.cli --help` before you write anything new: rebuilding an
instrument that already exists is this repo's #1 failure mode.

Run commands as `python -m vqlab.cli <cmd>`, or with `PYTHONPATH=src` from the
repo root when vqlab is not installed in the active interpreter.

Commands are grouped by stage: `vqlab <namespace> <cmd>` (`vqlab fit moe`).
`vqlab <namespace>` lists one namespace, `vqlab --help` lists them all, and
`<namespace> <cmd> --help` shows a command's full surface. Every pre-namespace
flat name still works unchanged (`vqlab fit-moe` is `vqlab fit moe`, and queue
files may use either); `vqlab --help` shows each flat alias that differs.

## Pipeline stages

Stages run in the order listed; each stage is a CLI namespace. Every stage reads an artifact directory or a
teacher, writes a new directory (never in place), and leaves a record.

| # | stage | question it answers | commands |
|---|---|---|---|
| 1 | **plan** | What will it cost? Where should the bits go? What IS this model? | `plan onboard` (sequence a new teacher: profile -> caches -> determinism -> init sweep), `plan family-profile` (headers only: legal (d,K), GiB per bit, module signatures), `plan price`, `plan zero-groups` (code bytes spent on the teacher's near-zero rows; 397B ~10%), `plan layer-leverage` (rank by the traj_rel JUMP), `plan loo-bands` (restore one band to exact per hybrid, score in the assembled network: the damage map), `plan alloc-sweep`, `plan probe-init`, `plan preflight-ram`, `plan preflight-disk`, `plan mtp-probe` / `plan mtp-probe35` |
| 2 | **fit** | teacher weights -> codebooks + codes | `fit fits` (find / reuse stored fits FIRST), `fit moe`, `fit dense`, `fit ple`, `fit geo-build` (refit named modules, keep the rest), `fit additive` (two small codebooks expanded to one ordinary K1*K2 fit: a KL test with no new kernel), `fit reselect` (re-pick codes under a self-generated activation Gram, codebook fixed; F67-F78, re-test under the KL gate), `fit harvest-parts` |
| 3 | **build** (`assemble/`) | codes -> an artifact that loads | `build pack`, `build pack-dense`, `build pack-ple`, `build splice-ple`, `build ple-swap`, `build unpack-dense`, `build reskeleton`, `build slice` (a small REAL slice of any teacher, e.g. `--layers 0-3`: preflight a new writer on it before the full run), `build minibase` (a band's shards as a fit base) and `build mix` (per-layer-band sources -> one artifact, shard by shard), `build stream-convert`, `build teacher-prep` (an official release -> an exact teacher in one command: relabel + `build sanitize-stream`, one layer at a time), `build dense`, `build graft`, `build graft-extras` (DeepSeek Vision-Exp tower + image routing from the release), `build mtp-extract` / `build mtp-pack` / `build mtp-graft`, `build mtp-head-ds4` (DeepSeek-V4-Flash's head), `build mtp-arms` (pinned per-head arms for `bench speed-pair-knurlogic --draft`) |
| 4 | **bundle** | Ship the runtime inside the artifact | `bundle moe` (MoE), `bundle dense`, `bundle patch-arch`, `bundle vision-layout` |
| 5 | **gate** | Is it loadable, correct, and releasable? | `gate check`, `gate check-release`, `gate check-bundle`, `gate bundle-accept`, `gate verify`, `gate smoke`, `gate vision-smoke`, `gate check-comparator`, `gate parity` (a family's checklist of reference-inference behaviours, each with its test; F195), `gate selftest`, `gate validate` (overnight queue), `gate pin` (freeze + smoke a copy before measuring), `gate mtp-smoke-head`, `gate spelling` (US spelling in released text) |
| 6 | **score** | How much damage does it carry? | `score kl-ladder` (the release gate), `score kl-pair` (pair two runs after the fact, zero GPU), `score kl`, `score ppl` (ppl, printed, not gated), `score stream-score` (the layer-streamed scorer kl-ladder runs; also builds teacher caches), `score tasks` (task benchmarks), `score kernel-truth`, `score kernel-truth-moe`, `score act-stats` (how often each clamp in a family's parity checklist fires on the house corpora), `score runtime-equiv` (two interpreters, same slice: bitwise or not) |
| 7 | **bench** | How fast is it, and where does the time go? | `bench decode-timeline`, `bench prefill-timeline`, `bench decode-ladder`, `bench active-bytes`, `bench stage-bandwidth` (GB/s per decode stage, time above the roofline), `bench serve-timeline` (a Knurlogic-served request, partitioned), `bench gpu-capture` (one decode step as an Xcode GPU trace: per-kernel counters), `bench prefill-bench`, `bench coverage`, `bench host-attrib`, `bench hc-micro`, `bench mtp-bench`, `bench mtp-accept`, `bench speed-pair` (two arms, fresh process each, ratio), `bench speed-pair-knurlogic` (same, served by Knurlogic: one Mac or a pipeline split) |
| 8 | **ship** | Publish or serve it | `ship publish` (a human action), `ship serve`, `ship mtp-generate` |

Skip-zero, `src/vqlab/skipzero/` (a shipped format since 2026-09-29, served natively by the runtime): `build sz-pack` / `gate sz-check` / `build sz-resident` / `gate sz-bitexact` (vq-skipzero: fully-dead VQ rows dropped on disk; 397B -10.8% disk / -11.1% resident at identical KL; docs/SKIPZERO.md).

Around the pipeline (`ship`, and `lab` for the lab itself): `ship release-prep` (sizes, junk, provenance, gate, Hub diff, then the exact `ship publish` line), `ship size` (text / +tower / +MTP, the card's three numbers), `ship card` (the whole model card from config + KL JSON + build record; refuses numbers that are not this artifact's), `ship card-tables` (the card's KL table from the scorer's JSON), `lab doctor` (interpreter, mlx builds, storage, token: run it first on a new machine), `lab config` (where vqlab reads and writes; `lab config init` on a new machine), `lab scratch` (`lab scratch reclaimable`: unpacked dirs whose verified pack exists, with sizes and the rm lines; never deletes), `lab queue` (run a list of steps from pinned code under the GPU lease; `--preflight` first; `status` shows an ETA from the data; `--on BOX` checks paths read on that box first), `lab reserve` (who has a box until when; `lab queue run` refuses a box reserved for someone else), `lab mcp` (the lab over MCP for agents), `lab gui` (read-only local window).

## Records: nothing happens without one

| record | where | what | read with |
|---|---|---|---|
| **run log** | `~/.vqlab/runs.jsonl` (per user, every `vqlab` call) | argv, commit, dirty files, library versions, exit code, duration | `vqlab lab runs` |
| **score stamp** | the `measured` block in every scorer's output | which artifact a number is OF: build-record id, pin, runtime md5/mtime, byte fingerprint, run id | read the score's JSON |
| **build record** | `<artifact>/vqlab_provenance.json` | how these bytes were made: fitter settings, inputs + lineage, per-module origin, runtime profile, hashes | `vqlab lab provenance <artifact>` |
| **tamper stamp** | `manifests/` (outside the artifact) | were the shard bytes rewritten? (for artifacts built before build records existed) | `vqlab lab manifest check` |
| **fit store** | `<store>/fits/<family>/<teacher>/L<layer>/<proj>/d<D>-K<K>/<fit_id>` + `index.jsonl` (HDD archive; `$VQLAB_FIT_STORE`) | every fitted module with its recipe; fitters file into it, `fit geo-build --pool` reuses from it (recipe must match) | `vqlab fit fits list / stats / census / retag` |
| **artifact registry** | `registry/artifacts.jsonl`, `registry/hub.jsonl` (in git) | every artifact from provable facts (runtime, geometry, size, fingerprint, record id); latest Hub comparison | `vqlab lab registry list / hub` |
| **queues** | `~/.vqlab/queues/<stamp>-<name>/` | each step's cmd, stdout, stderr, verdict; the pinned commit | `vqlab lab queue status / list` |
| **box reservations** | `<scratch>/vqlab-reservations.json` (the shared SSD; `$VQLAB_RESERVATIONS`; local fallback `~/.vqlab/reservations.json`) | who has each box until when, and why | `vqlab lab reserve --list`, MCP `gpu_state` |
| **family data** | `families/<family>/` | `entry.json` (how to read the tensors, no code change) and `teachers/<teacher>/{profile,onboard,teacher_caches}.json` | `vqlab plan family-profile`, `vqlab plan onboard`, MCP `where_is` |
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
changes what `gate check-bundle` compares every published artifact against.
