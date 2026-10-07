# The checklist: a new family, a new rung, a release

One list, in order, so nothing is left out by accident. Each line names the
command that does it and the document that explains why. A line is ticked
by its command's output, recorded where the step says, never by memory.
Copy this list into the campaign's notes and tick it there.

## A. A new model family (before any fit)

- [ ] `vqlab onboard --teacher <dir>`: profile, loader check, cache
      determinism, init sweep. ONBOARDING.md says why each step exists.
- [ ] Family entry in `core/families.py` (with `model_type`, so
      artifact-only tools resolve it); the CPU loader gate passes:
      `python src/vqlab/core/expert_src.py --selftest`.
- [ ] Teacher archived in configured storage, never re-downloaded
      (`vqlab config`, `where_is`).
- [ ] Depth law MEASURED for this family (`layer-leverage`, ranked by the
      jump in `traj_rel`), never inherited from another family.
- [ ] **Speed baseline of the bf16 or reference build** (section D), so
      every rung has something to be a ratio of.

## B. A new rung (fit, assemble)

- [ ] Priced before fitting (`vqlab price`); `K * dim * 2 < 32768`
      checked for the fused kernels (FINDINGS IV).
- [ ] F-number reserved (`next_f_number reserve=true`); predictions
      pre-registered in it.
- [ ] Fit with the per-geometry `--relerr-abort` (FINDINGS I.8).
- [ ] Bundled; `vqlab check-bundle`, `vqlab verify`.

## C. Quality gate

- [ ] `vqlab kl-ladder`, paired, three corpora at 12288, |t| > 2 (the
      release gate since F118). Ppl printed with its sign, not gated.
- [ ] Comparators passed `check_comparator.py` before their rows are
      believed.

## D. Speed card (every rung and every family's reference)

Recorded, not gated: until each family has several cards there is no
measured threshold to gate on, and a threshold nobody measured is a
guess dressed as a gate. MEASUREMENT.md has the method and what each
instrument cannot see. Pinned copy, smoked, quiet box, machine and peak
bandwidth stated.

- [ ] `vqlab bench decode-timeline --art <pin> --json-out dt.json`:
      verdict SERIAL or ESTIMATOR GAP, noise floor stated.
- [ ] `vqlab bench stage-bandwidth <pin> --timeline dt.json --peak-gbs <peak>`:
      overall % of peak; the top three stages by excess ms; no unclaimed
      bytes (unclaimed bytes are a mis-named or mis-wired tensor: fix
      before release, it is a correctness finding, not a speed one).
- [ ] `vqlab bench prefill-timeline --art <pin> --tokens 2048`: drift
      stated; the top stage types.
- [ ] Served through Knurlogic: `vqlab bench serve-timeline --prompt-tokens
      2048 --n 3`: TTFT, prefill and decode tok/s, the engine-only prefill
      rate, `spans_unaccounted_s` ~ 0.
- [ ] Decode and prefill as a RATIO against the family's reference, same
      session (`vqlab bench speed-pair` / `speed-pair-knurlogic`), never an
      absolute (FINDINGS III).
- [ ] The card's numbers saved as JSON beside the campaign's finding.

## E. Release

- [ ] `vqlab smoke --max-tokens 8` and `vqlab vision-smoke` on the bundle
      that ships; one token through the shipping runtime (rule III.11).
- [ ] `vqlab check-release`; `vqlab selftest`.
- [ ] Baseline is the HF revision, not the local copy (AGENTS.md
      "Releasing?").
- [ ] Card written from the artifact's own `config.json`, never from a
      narrative source; it carries the speed card (D) with its machine.
- [ ] Publish (a human's action).

## F. Close the campaign

- [ ] Finding written (`findings_append`); fits filed in the fit store.
- [ ] `vqlab lab scratch reclaimable`: hand the `rm` lines to Noah.
