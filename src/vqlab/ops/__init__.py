"""vqlab.ops -- the lab operations on THIS box, as a library.

What `vqlab.api` and every MCP server (ours in `vqlab.agents.mcp`, or a
developer's own) are built on. Plain data in, plain data out; a refusal
raises `LabError` (errors.py), which `vqlab.api` exports as `ApiError`.

    errors.py      LabError: the one refusal type (stable `code` + JSON `extra`)
    roots.py       the configured storage roots, path guards, directory walk
    storage.py     read-only: where_is, list_artifacts, artifact_config, disk_free
    docs.py        read_doc: a slice of a lab document (never source code)
    runs.py        launch (allowlisted, detached, GPU lease), status, stop, list_runs
    supervisor.py  the detached process a launch starts: holds the lease, retries
    box.py         gpu_state: lease holder, exo placements, reservations, other boxes
    queues.py      queue_status: `vqlab queue` runs from their state.json
    findings.py    next_f_number (reserve), release_f_number, findings_tail,
                   findings_append -- the only tool-side writer of the log

Importing this package imports nothing; each module is loaded on use.
(Named `ops`, not `lab`: `lab/` is the gitignored private notebook.)
"""
