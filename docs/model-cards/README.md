# Model cards — ready to publish

The 20 `.md` files in this folder are the corrected cards, one per Hub repo,
each named exactly after its repo. They are finished except for one thing:
**the entry for the new runtime.**

## Add the runtime entry

Every card has a `## Changelog` with dated entries, newest first. Entries
accumulate — do not overwrite an older one.

```bash
cd docs/model-cards
python3 _tools/add_changelog_entry.py \
    --date 2026-09-09 --label "runtime <version>" --body entry.md --cards .
```

- `--body` = a markdown file with just the entry text, no heading.
- `--only <prefix>` restricts to one family, e.g. `--only Qwen3.8-27B`, so
  per-family wording is one run each.
- `--dry-run` previews. Re-running the same date+label is a no-op.
- The three GLM cards have no prior entry; the tool creates the section.

## Then publish

The publish source is the artifact's own `README.md` — not this folder, and
not `research/quantlab/MODEL_CARD_*.md`. Copy, then push:

```bash
A="/Volumes/Thunderbay SSD/Exo Models/TheDrainFlorist--<name>"
cp <name>.md "$A/README.md"
python -m vqlab.cli publish --artifact "$A" --repo TheDrainFlorist/<name> \
                            --files model.py README.md
```

`model.py` in the push means the full gate runs, generation smoke included.
397B and GLM rungs need the cluster smoke (M4 publishes, M3 master API).

As of 2026-09-09 every artifact `README.md` is still byte-identical to what
is on the Hub, so nothing has been clobbered.

## House rules these cards follow

Seven sections on every card, same names, same order:

`Measured results` → `Run it` → `Methodology` → `Verification` →
`Limitations` → `Paper` → `Provenance`

Conditional sections keep a fixed slot: `Requirements`, `Changelog`,
`Runtime`, `Memory`, `Speculative decoding (MTP)`, `Hardware`, `Vision`,
`Siblings`, `Comparators`, `Task benchmarks`.

- H1 is the bare repo name — the Hub already shows the owner.
- Provenance is mandatory: base model, its licence, the derivative
  statement, the quantization attribution.
- Paper links use the concept DOI `10.5281/zenodo.22119017`, never a version
  DOI (see `research/quantlab/handoff/SESSION_PAPER.md`).
- No `TODO` / `TO MEASURE` / `[PENDING]` markers.

## Before you publish

```bash
python3 _tools/check_cards.py .
```

Fails on: missing sections, missing licence/attribution, wrong H1, unbalanced
fences, placeholder markers, a version DOI, and **any `~` outside code**.

A bare `~` (as in "~2%") is a strikethrough delimiter on the Hub: two of them
in the same block strike out everything between. Write `≈2%`, or drop the
tilde where a word like "up to" already hedges. `_tools/fix_tildes.py` does
this in bulk.

## Folders

- `_published-2026-09-09/` — snapshot of the live Hub cards, for diffing.
- `_tools/` — the pipeline that produced these from that snapshot, plus
  `CARD-CORRECTIONS.diff` (every change, per card).

## Note

`research/quantlab/make_flashnext_cards.py` and `make_qwen38_cards.py` still
emit the old structure. Fold these rules in before using them again.
