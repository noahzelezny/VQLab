# VQ Lab Bench — GUI mockup (2026-09-29, not wired)

Static, self-contained mockup of the next `vqlab gui`: Library · Recipe · Jobs.
Every number on the Library page is `research/paper/RESULTS-V5.md` parsed into
`results.json` and embedded; the Recipe estimates and the
per-module time model are illustrative until wired to `/api/*`, the fit store,
`price`, and `queue status`.

Serve it:  `python3 -m http.server 8799 --bind 127.0.0.1 --directory src/vqlab/agents/gui_static/bench`
(`index.html` is `bench.html` wrapped in a doctype; edit `bench.html`, regenerate
`index.html` by wrapping it, or just serve `bench.html` from the artifact host.)

Published copy: https://claude.ai/artifact/HxsGKe2CcweEWAy1hM2p8P

Jobs page: real `vqlab queue` runs. `vqlab gui` serves this page at `/bench/` and the
Jobs page reads `/api/queues` + `/api/runs` live (10 s refresh); opened as a file it
falls back to the embedded `JOBS_SNAPSHOT`. Refresh it with `python make_jobs_snapshot.py`
(writes `jobs.json`, patches `bench.html`, regenerates `index.html`).
