# selftest/ — the whole pipeline, on a tiny synthetic model

Layer L12 (tests/test_layering.py): it drives every stage below it, so it
may import any of them, and nothing below may import it. Command:
`vqlab selftest` (also `vqlab gate selftest`). Was gate/selftest.py until
untangle step 14; `vqlab.gate.selftest` and the bare `selftest` still import
this package for one release (DeprecationWarning).

## Inputs
None: it synthesizes its own checkpoints in a temp workspace and points the
fit store, manifest dir, families dir and queue dirs at it. Only the
reselect fixture writes under configured scratch (and removes it).

## Process
- `__init__.py` — `main` (the runner: stages in print order), `check` /
  `skip` / `run` and the PASS/FAIL/SKIP report, the `Ctx` stages share.
- `synthetic_model.py` — the tiny dense checkpoint, the geo-build and
  fit-moe MoE fixtures, the unknown-family teacher, `decode_all`.
- one module per stage checked: `fit_checks`, `gate_checks`,
  `build_checks`, `records_checks`, `kernel_checks`, `family_checks`,
  `skipzero_checks`, `plan_checks`, `queue_checks`, `layout_checks`.
- tools' own self-checks, moved out of the tools: `reselect_fixture`
  (`vqlab reselect selftest`), `sz_check_fixture` (`vqlab sz-check
  --selftest`), `sz_bitexact_cpu` (`vqlab sz-bitexact --selftest`). The CLI
  routes those flags here (`vqlab.commands.SELFTEST_ROUTES`), since a tool
  cannot import upward.

## Outputs
PASS/FAIL/SKIP lines and `N passed, N failed, N skipped`; exit 1 on any
FAIL. Adding a check: put it in its stage's module and keep the print
order; the line text is what users and CI compare.
