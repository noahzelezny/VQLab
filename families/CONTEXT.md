# families/ — everything VQLab knows about each model family, as data

Broad to specific: `families/<family>/` for what holds across the family,
`families/<family>/teachers/<teacher>/` for one checkpoint.

| file | written by | what |
|---|---|---|
| `<family>/entry.json` | `vqlab family-profile --write-entry` or by hand | how to READ the family's tensors (src key template, projections). A data entry: no code change. Code entries in `src/vqlab/core/families.py` win on a name clash. |
| `<family>/teachers/<teacher>/research/` | the lab, by hand | that teacher's ledgers, sweep plans, defect notes and result tables (moved from research/quantlab on 2026-09-25) |
| `<family>/teachers/<teacher>/profile.json` | `vqlab family-profile --teacher <dir>` | header-only census: arch, bytes by class and per layer, GiB per bit, module shape signatures, every legal (d, K) and its exact packed size |

Onboarding a new family: `vqlab family-profile --teacher <dir>`. If it says
UNKNOWN, review the drafted entry and rerun with `--write-entry <name>`.
Then follow docs/ONBOARDING.md for the measured steps (init sweep,
leverage, allocation). Measured results are appended to this folder, never
kept only in prose.
