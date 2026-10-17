# queue/ — Queue: run lists of commands, overnight, from pinned code

Layer L11 (tests/test_layering.py). It launches commands BY NAME through
`vqlab.commands` (`python -m vqlab.cli <cmd>`) and never imports a tool
module, agents/, selftest or cli. The GPU lease and the exo check it shares
with the MCP server live below it, in `core/box/gpu_lease.py` and `core/box/exo.py`.

## Inputs
A queue file (JSON: `name`, `commit`, `python`, `steps` of `cmd`|`script`,
`args`, `expect`, `preflight`, ...; full format in `runner.py`'s docstring);
`validate`'s artifact list (`caches/validate/queue.jsonl`).

## Process
- `runner.py` — `vqlab queue run|status|list|wait|stop` (CLI `main`); `run()` is
  the runner process: take the GPU lease, drain the steps, tidy preflight
  outputs and the pinned worktree. `run_step()` checks then runs one step
  with retries.
- `steps.py` — the step schema (`validate`), the args a step runs with
  (`preflight_args`, `redirect_outs`), the checks before it may run
  (`static_check`: input paths exist IN THE PINNED TREE; `pin_check`:
  scoring steps refuse an unsmoked pin), its argv and env.
- `files.py` — queue dirs (`queues_dir()` = `config.queues_dir()`),
  state.json (`load_state`, `save_state`), `create()` (pin a worktree, or
  an installed package by version + hash). Schema string `"vqlab.queue/1"`
  is data in every state.json: never rename it.
- `report.py` — `status` (with ETA), `list_queues`, `wait` (judged from
  state.json; a dead runner is terminal).
- `remote.py` — `queue run --on BOX`: reservation, remote path READ check
  over ssh, fast-forward the box's clone, launch detached there.
- `eta.py` — ETAs from progress lines or past runs of the same step.
- `reserve.py` — `vqlab reserve`: who has a box until when; `queue run`
  refuses a box reserved for someone else.
- `validate.py` — `vqlab validate`: the overnight validation queue that
  drains artifacts through the gates (by command name), never publishes.

## Outputs
`<records>/queues/<stamp>-<name>/` (queue.json, state.json,
steps/NN-name/{cmd,stdout,stderr,verdict.json}, tree/ while running).

## Rules that bite
- Never a bare `queue` alias, and no bare alias for queue/'s generic
  stems (`_layout` asserts it): `import queue` is the stdlib's in every vqlab process
  (tests/test_queue_package.py). Always `vqlab.queue.<module>`.

## Files
- `eta.py`: ETAs for `queue status` / MCP queue_status, from the data
- `files.py`: queue files on disk: where queues live, state.json, creating one
- `remote.py`: `queue run --on BOX`: launch on another box after checking every path reads there
- `report.py`: `queue status | list | wait`: read queue state, never change it
- `reserve.py`: `lab reserve`: who has a box until when, visible to every box
- `runner.py`: `lab queue`: run a list of steps from pinned code, under the GPU lease, failing loudly
- `steps.py`: queue steps: the schema, the arguments a step really runs with, pre-run checks
- `validate.py`: `gate validate`: drain artifacts through the release gates overnight; never publish
