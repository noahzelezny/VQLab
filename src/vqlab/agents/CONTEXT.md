# agents/ — Agents: the lab over MCP

## Inputs
JSON-RPC over stdio (`mcp`). (Queue files: `queue/`, untangle step 12.)

## Process
`vqlab mcp`: `where_is`, `list_artifacts`, `run` (allowlisted, detached, under the GPU lease), `next_f_number` (reserve=true before a long run), `findings_append`, ...

The server is the package `mcp/` (untangle step 13), and since step 16b it is a THIN ADAPTER: each tool is a JSON schema plus an argument mapping onto `vqlab.api` (or `vqlab.ops` where the api deliberately has no call: `run` returns the launcher's raw record, `artifact_config`, `read_doc`, `queue_status`). No lab logic lives here; the implementations are in `src/vqlab/ops/` (see its CONTEXT.md). `mcp/server.py` (stdio JSON-RPC loop + the tool table: name -> fn, schema; `--supervise` delegates to `vqlab.ops.supervisor`), `mcp/common.py` (`ToolError` = `vqlab.ops.errors.LabError` = `api.ApiError`, old helper names), `mcp/tools_storage.py`, `mcp/tools_runs.py`, `mcp/tools_findings.py` (the `t_*` adapters; old module-level names re-exported). `mcp_server.py` is a deprecated RE-EXPORT of every old name. Monkeypatching a name on a re-export does not reach the code: patch the defining `vqlab.ops.*` module.

`vqlab scratch reclaimable` (`scratch.py`, was `vqlab.records.scratch_cmd`, deprecated alias for one release): unpacked dirs whose packed counterpart's build record names them and still verifies; prints the rm lines, never deletes.

`vqlab queue run <file>` (code in `queue/`, see its CONTEXT.md; `run_queue.py` here is a deprecated re-export of the old names): a multi-step campaign (a night of scoring) run from a git worktree PINNED at one commit, holding the same GPU lease as `run`, retrying resumable builds, refusing unsmoked pins, and failing a step on a nonzero exit, a traceback, or missing output. `--preflight` runs every step's small real version first; `--detach` survives the shell. `queue status` (and MCP `queue_status`) prints an ETA from the running step's own progress lines (fit-moe `[i/N] shard (Ts)`, kl-ladder cells) or past passed runs of the same step, else "ETA unknown". `--on BOX` first checks over ssh that every path the queue names exists AND reads on that box (an SMB mount can list while reads fail EIO), and refuses with the fix ("remount <volume> on <box>").

`vqlab reserve` (`queue/reserve.py`; old `vqlab.agents.reserve` is a deprecated alias) `BOX|here --for WHO --until +3h|HH:MM [--note]`, `--list`, `--release`: intent, not a lock. Stored in `<scratch>/vqlab-reservations.json` (the shared SSD every box mounts; `$VQLAB_RESERVATIONS` overrides; falls back to `~/.vqlab/reservations.json`, local-only, when scratch is unreachable). Expired ones are ignored. MCP `gpu_state` lists every box's; `queue run` / `queue run --on BOX` refuse a box reserved for someone other than `$VQLAB_WHO` (else login) unless `--override-reservation`, which state.json records.

`[boxes.NAME] teachers = "<path>"`: that box's local teacher copies. `config.teachers()` returns it when running ON that box (`$VQLAB_BOX`, which `--on` sets, else hostname), and `vqlab <cmd>` expands a `<teachers>/X` argument through it, so a queue step naming `<teachers>/Model` reads the local copy there.

`vqlab config` / `config set KEY PATH` / `config init` and the GUI's Settings view share ONE implementation, `agents/settings.py` (`locations`, `validate`, `set_value`, `campaign_sizes`): validate (absolute, a directory or creatable with `--create`, writable, free space shown), then rewrite only that key's line in `[paths]`. Every location defaults under the lab folder `lab` (`$VQLAB_HOME`, `[paths] lab`, else `~/VQLab`): `<lab>/models`, `scratch`, `fits`, `records`, and `teachers` (bf16 teachers) defaults to the models folder; with no `lab` set, data already under `~/.vqlab` or `~/.local/share/vqlab` keeps resolving there. `usage()` (GUI `/api/storage/usage`) groups the locations by volume with what each holds. `records` (`$VQLAB_RECORDS`, else `<lab>/records`) holds user-written state: `FINDINGS-LOG.md` (`$VQLAB_FINDINGS_LOG` still overrides), `registry/`, `families/` (`$VQLAB_FAMILIES_DIR`), `research/`, `queues/` (`$VQLAB_QUEUE_DIR`). Accessors: `config.records()`, `findings_log()`, `registry_dir()`, `families_dir()`, `research_dir()`, `queues_dir()`. A checkout that wants the old in-repo `lab/` sets `records` to it.

## Outputs
Detached runs under the MCP runs dir, FINDINGS-LOG entries; queues under `<records>/queues/<stamp>-<name>/` (state.json, steps/NN-name/{cmd,stdout,stderr,verdict.json}).

## Rules that bite
- Use `where_is` before claiming anything is missing.
- `run` launches through `python -m vqlab.cli`, so every agent job is in the run log.
- A queue step never imports the live checkout: commit mid-run freely. Uncommitted src/ changes are NOT in the queue (it refuses unless --allow-dirty).
- Every queue step needs a `preflight` block; run `--preflight` before any multi-hour queue.
- `onboard.py` (`vqlab plan onboard`) lives here, not in plan/: it drives the MCP server's `t_run`/`t_status`. Old name `vqlab.plan.onboard` is an alias for one release. The GPU lease file logic is `core/box/gpu_lease.py`.

## Files
- `config_cmd.py`: `vqlab lab config`: show, set or initialise storage locations (over settings.py)
- `doctor.py`: `vqlab lab doctor`: which interpreter, packages, storage and credentials a run would use
- `model_scan.py`: `vqlab lab scan`: models already on this machine (vqlab's own roots, the Hugging Face cache, LM Studio, exo, Ollama, `[scan] extra`), tagged original / quantized / incomplete / unreadable with their vqlab family, into `<records>/registry/found.jsonl`; reads config.json and file lists only. The GUI rescans at launch and on Rescan; first run scans once
- `first_run.py`: on a new install (no config, no location env, no older data) the first `vqlab` command creates `~/VQLab/{models,scratch,fits,records}` (bf16 teachers default to models/) and a config naming it, and says so once; `$VQLAB_NO_SETUP=1` skips it
- `gui.py`: `vqlab lab gui`: serves the VQ Lab GUI (Library · Recipe · Jobs · Downloads · Storage: gui_static/index.html, built from gui_static/bench/bench.html) and the JSON it reads
- `gui_static/bench/make_jobs_snapshot.py`: regenerates the GUI's embedded jobs snapshot (bench.html / index.html)
- `mcp/server.py`: `vqlab lab mcp`: the stdio JSON-RPC loop and the tool table (name -> fn, schema)
- `mcp/common.py`: `ToolError` (= `vqlab.ops.errors.LabError`) and old helper names, re-exported
- `mcp/tools_storage.py`: adapters: where_is, list_artifacts, artifact_config, read_doc, disk_free
- `mcp/tools_runs.py`: adapters: run, status, stop, list_runs, gpu_state, queue_status
- `mcp/tools_findings.py`: adapters: next_f_number (reserve / release), findings_tail, findings_append
- `mcp_server.py`: DEPRECATED re-export of every old MCP name (one release)
- `onboard.py`: `vqlab plan onboard`: sequence a new teacher through docs/ONBOARDING.md
- `run_queue.py`: DEPRECATED re-export of the queue runner's old names (now `vqlab.queue`)
- `scratch.py`: `vqlab lab scratch`: what on the scratch volume a human could delete, and why
- `settings.py`: storage settings: read, validate and write vqlab's locations (CLI and GUI share it)
