# research/ — per-model-family arcs

One directory per model family under active investigation. Each carries a
LEDGER.md: the family's arc ledger, dated entries, newest wins — same
conventions as paper/LEDGER.md, scoped to the family.

Division of authority (unchanged by this structure):
- FINDINGS.md (root)      cross-family LAWS and retractions. A result that
                          generalizes gets promoted there; family ledgers
                          hold the family-specific record.
- EXPERIMENTS.md (root)   numbered experiments, cross-family, append-only.
- paper/LEDGER.md         the published paper's book: its versions, its
                          claims, its corrections. Closed to new-arc
                          material now that research/ exists.
- research/<family>/      everything else about the family: rung tables,
                          tooling incidents, readiness docs, card drafts.

Commit as you go — an uncommitted ledger entry protects nothing.
