# VQ Lab GUI

`vqlab gui` serves this page at `/`: Library · Recipe · Jobs.
Every number on the Library page is `research/paper/RESULTS-V5.md` parsed into
`results.json` and embedded; the Recipe estimates and the
per-module time model are illustrative until wired to `/api/*`, the fit store,
`price`, and `queue status`.

Serve it:  `python3 -m http.server 8799 --bind 127.0.0.1 --directory src/vqlab/agents/gui_static/bench`
(the shipped page is `../index.html`: `bench.html` wrapped in a doctype; edit `bench.html`, regenerate
`../index.html` by wrapping it, or just serve `bench.html` from the artifact host.)

Jobs page: real `vqlab queue` runs. `vqlab gui` serves this page at `/` and the
Jobs page reads `/api/queues` + `/api/runs` live (10 s refresh); opened as a file it
falls back to the embedded `JOBS_SNAPSHOT`. Refresh it with `python make_jobs_snapshot.py`
(writes `jobs.json`, patches `bench.html`, regenerates `../index.html`).
