# PROVENANCE — every artifact carries a record of how it was made

**Status: PROPOSAL, first slice implemented (2026-09-25). Awaiting Noah's
review before the broad rollout (§6).** Nothing here rewrites a published
artifact or touches the storage array artifact roots.

## 1. Why

The paper v5 revision (2026-09-23..25) had to reconstruct, by reading
EXPERIMENTS.md E92-E118 by hand, facts that no artifact recorded:

| # | what was lost | example |
|---|---|---|
| 1 | **fitter vintage** | 397B 2.4bpw fit by the 08-16 fitter (random init, one-hot k-means; `fitter_0816_cdcdeab.py`), 2.6/3.1 by the 08-18+ k-means++ fitter, and layers 57-59 of each corrected rung refit by geo-build, whose default adds scale<->codebook alternation (F80/F81) the others lack. Three recipes in one family, one of them *inside* each rung. |
| 2 | **runtime profile** | 35B 3.4 ships v1.5, 3.8/4.6/5.4 ship v2; the 397B rungs carry three bundles (md5 151f1eda / ca3645e0 / 55d1e38e); 27B published vs rebuilt: 545d4810 vs 50cdc89e. |
| 3 | **local vs Hub drift + in-place rebundles** | 8/20 `Exo Models/` copies differed from HF (2026-09-19); F151 lost 3 of 13 runs to a concurrent rebundle. |
| 4 | **the paper's manifest claim** | The paper says published artifacts "carry external manifests". `vqlab manifest` writes a *tamper stamp* (shard size / mtime / sha256 of the first 1 MiB) into a cwd-relative, gitignored `manifests/` dir. It records nothing about how bytes were made, and no stamps for the published fleet were found. **The claim should be softened in DRAFT.md until §6 step 4 lands.** |
| 5 | **"same recipe" != "same bytes"** | A rebuilt 35B struct base with the same quantization map differed from the published one (scales up to 12.8%) after an mlx version change. |
| 6 | **defaults that change the method** | geo-build alternation on by default; geo-build `--seed` rejects -1 where fit-dense accepts it; fit-moe init default flipped 08-18. |

The common cause: the recipe lives in argv, module constants, the tool's
source at some commit, and the library versions — and none of it is written
next to the bytes.

## 2. What already exists (extend, don't duplicate)

| instrument | what it answers | role in this design |
|---|---|---|
| `vqlab manifest` (`artifact_manifest.py`) | were these shards rewritten since stamped? | **kept** as the cheap tamper stamp for artifacts that predate records; `provenance --verify` subsumes it for new builds. Help text renamed to say "tamper stamp". |
| `vqlab check-bundle` + `runtime_profile.py` | is the bundled `model.py` the repo runtime, under which profile? | **reused**: the record's `runtime` block calls `profile_of` / `flags_of`. |
| `check-release`, `validate` queue | release gates | future: `check-release` requires a record (§6 step 5). |
| `geo-build` part naming + `harvest-parts` `manifest.json` | fit identity by (module, d, K, tag) | the new `origins.json` ledger in the parts dir complements it with *how* each part was made. |
| `fit_ple.py` `ple_manifest.json` | per-tensor geometry + relerr for PLE fits | folds into `method`/`modules` when fit-ple adopts the writer. |
| MCP `where_is` / `list_artifacts` | where is it, what is it | future: `list_artifacts` returns each artifact's record id (§4). |

## 3. The build record — `vqlab_provenance.json`

Written by every build tool as its **last** step, **inside** the new output
directory (a build output is always a fresh dir, so the record cannot be
invalidated by the build itself; tools that copy a parent's side files skip
the parent's record). Schema `vqlab.provenance/1`:

```
id        sha256 of the record's canonical JSON (minus id) — content address
created   UTC timestamp
tool      name, exact argv, cwd, args.resolved, args.at_default   ← problem 6
code      repo, commit, dirty + dirty_files, script sha256         ← problem 1
env       python, mlx, mlx_lm, numpy, host, platform               ← problem 5
method    the tool's fitter settings, INCLUDING module constants    ← problem 1
          (init, lloyd iters, sample size, alternation + rounds, scales,
           tail weighting, seed, rng)
inputs    [{role: teacher|base|reuse|source, path, realpath,
            provenance_id?, hf_repo?, hf_revision?, files{config,index,
            model.py: sha256}, shards{bytes, head_sha256}}]         ← lineage
modules   {module: {dim, k, origin: fit|reuse|resume, from?, fitter?,
           commit?, sha256, source_origin?, restored}}              ← problem 1
runtime   model_py sha256 + md5, profile v1.5|v2|custom, all VQ_* flags ← 2
outputs   {file: {bytes, sha256 | head_sha256, symlink_to?}}        ← 3, 5
```

Design choices, each with its reason:

* **Inside the artifact, not beside it.** It must travel to the Hub and to
  the M4's internal copies; an external manifest is exactly what went
  missing (problem 4). The old tamper stamp lives outside because stamping
  must not alter the bytes it describes — the build record avoids that by
  being written once, last, and excluding itself from `outputs`.
* **Lineage by content id, not path.** A child records its parent's
  `provenance_id`; a path can be rebundled under you (F151), an id cannot.
  Parents without a record (the whole current fleet, all teachers) are
  fingerprinted instead — shard sizes + head hashes, full sha256 of config /
  index / model.py, HF revision when the path is a hub snapshot.
* **Per-module origin, persisted across resumes.** geo-build writes
  `origins.json` in its parts dir as each part is fit or reused, so a resumed
  build still knows that module X was fit here at commit C with alternation
  on, and module Y was harvested from rung Z (whose own ledger entry is
  carried as `source_origin`). This is the record that would have made
  problem 1 a lookup.
* **Method includes constants, not just flags.** `ITERS_LLOYD`, `NFITG`,
  `ITERS_ALT` are module constants in geo-build; reading them requires the
  source at the right commit. The tool writes them out.
* **`at_default` is recorded.** When a default flips, older records say they
  depended on the old one.
* **Hashing cost.** Files this run wrote get full sha256 (up to
  `VQLAB_PROV_FULL_HASH_MAX`, default 8 GiB); carried-over symlinked shards
  and input shards get size + first-MiB sha256 (the existing manifest's
  scheme). A teacher is fingerprinted, not re-hashed.

## 4. The registry (proposed, not built)

One queryable index, `registry/artifacts.jsonl` **in git** (small, reviewable,
diffable), one line per known artifact instance:

```
{id, name, role: published|serving|candidate|base, path, host,
 hf_repo, hf_revision, parents:[id], runtime_md5, profile, text_bytes}
```

* `vqlab provenance --index <root>` scans a root, reads records, and appends
  / updates lines (read-only on artifacts).
* `vqlab provenance --backfill <artifact> --from-notes <yaml>` writes a
  record for a **pre-record** artifact into the *registry only*, never into
  the artifact dir — with `method` transcribed from EXPERIMENTS/FINDINGS and
  flagged `reconstructed: true, source: "E92-E118"`. That is where the
  397B fitter-vintage table from the paper revision should land first.
* MCP `list_artifacts` / `where_is` gain the record id + profile per hit.

## 5. Hub drift check (proposed, not built)

`vqlab provenance --hub <artifact> --repo <hf repo>`: fetch the repo's file
list with sizes + LFS sha256 from the Hub API (metadata only, no download),
compare against the local record or live fingerprint, and report per file
`same | local-only | hub-only | differs`. For `model.py` it also reports both
profiles. This mechanizes AGENTS.md "the baseline is the HF revision, not the
local copy" and turns the 8/20 drift audit into one command. Publishing
should record the resulting HF commit sha back into the registry.

## 6. Rollout plan (status 2026-09-25)

| step | scope | status |
|---|---|---|
| 1 | `provenance.py` writer + `vqlab provenance` (summary / --json / --verify / --lineage) | **done** |
| 2 | build records in `fit-dense`, `geo-build`, `fit-moe`; geo-build origins ledger | **done** |
| 2b | per-user run log (`~/.vqlab/runs.jsonl`, `vqlab runs`); run id stamped into records | **done** |
| 2c | fit store: every fit filed with its recipe (`vqlab fits`, `geo-build --pool`); HDD archive migrated | **done** |
| 3 | build records in `pack*`, `build-dense`, `bundle`, `rebundle-dense`, `stream-convert`, `graft`, `mtp-pack`, `splice-ple` | open (tracker VL4.7) |
| 4 | publish ships `vqlab_provenance.json`; paper DRAFT softens "external manifests" | needs Noah (VL4.10) |
| 5 | `check-release` fails an artifact with no record or a drifting `--verify` | after 3 (VL4.7) |
| 6 | registry + backfill for the published fleet (§4) | open (VL4.8) |
| 7 | Hub drift check (§5) | open (VL4.8) |
| 8 | defaults: seed 1234 everywhere, `-1` = explicit random | **done** (per-module geo-build seeding: VL4.9) |

## 7. Known gaps this record does NOT close

* **geo-build's RNG is one stream across modules.** A module's fit depends on
  which modules were fit before it in the same process, so a resumed build
  or a different geomap refits a module differently at the same `--seed`.
  The record makes this visible (`origin`, `commit`) but does not make it
  reproducible. Fix: seed per module from `hash(seed, module, d, K)`. That
  changes fits, so it is a method change and wants an A/B and a decision.
* **Records certify the bytes, not the numerics.** Problem 5 (mlx version
  moved scales 12.8%) is *recorded* by `env.mlx`, not prevented. A rebuild
  under a different mlx is a different artifact, and the record says so.
* **Content hashes are per file, not per tensor.** Two builds that differ in
  one tensor show as one differing shard. A per-tensor hash table (safetensors
  header offsets make it cheap) is a follow-up if shard granularity is too
  coarse for splice forensics.
* **Scores are not in the record.** A score describes an (artifact id,
  instrument) pair; the natural next step is for `kl-ladder` / `score` to
  stamp the record id they measured into their outputs, which would make
  AGENTS.md's "a comparison row must name the ARTIFACT" mechanical.
