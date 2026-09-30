# VQ Lab Bench — GUI mockup (2026-09-29, not wired)

Static, self-contained mockup of the next `vqlab gui`: Library · Recipe · Jobs.
Every number on the Library page is `research/paper/RESULTS-V5.md` parsed into
`results.json` and embedded; the Recipe estimates, job list, ETAs and the
per-module time model are illustrative until wired to `/api/*`, the fit store,
`price`, and `queue status`.

Serve it:  `python3 -m http.server 8799 --bind 127.0.0.1 --directory src/vqlab/agents/gui_static/bench`
(`index.html` is `bench.html` wrapped in a doctype; edit `bench.html`, regenerate
`index.html` by wrapping it, or just serve `bench.html` from the artifact host.)

Published copy: https://claude.ai/artifact/HxsGKe2CcweEWAy1hM2p8P
Design decisions and Noah's UI rules: memory file `function-over-form.md`.
