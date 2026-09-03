# Morning report — overnight of 2026-09-02 -> 03

## TL;DR
The fleet is push-ready. Two review items (cards, 397B sidecar placement)
and the runbook is docs/PUSH-RUNBOOK.md. LinkedIn draft:
docs/LINKEDIN-DRAFT.md — post after the push so links resolve.

## Shipped overnight (all local, nothing pushed)
- Release validation: 18/18 re-bundled with the final kernel, static
  gates PASS, full check-release PASS on 2.1bpw + 27B x3, referee
  identical to 16 digits, A-B-A +5.5% (18.09 -> 19.09 tok/s).
- Found + fixed 4 dense bundles that were SHIPPING BROKEN (would
  ModuleNotFoundError on stock mlx-lm); root-caused to the deleted
  quantlab-patched venv; gate now refuses to run from a patched venv.
- NO-SHIP: gemma-4-e4b-PLE (pre-existing Metal threadgroup defect).
- GLM memory story CLOSED: prefill footprint was the MLX buffer cache;
  EXO_MLX_CACHE_LIMIT_GB=6 pins it at 0.0-0.2G/chunk on both ranks,
  zero speed cost (134s for 26k tokens). Both boxes' env hooks set.
- exo MTP stage 0 smoked LIVE through exo serving: 25.66 tok/s,
  acceptance 0.76 (isolated node, Flash 2.1bpw).
- exo MTP stage 1: implemented + 22 unit tests on branch mtp-stage1
  (both checkouts synced to 6de2f466); crashed the cluster once via an
  incomplete rename (fixed, f7321f34). Cluster smoke NOT done: co-located
  2-node test fails placement on memory accounting; needs the real pair.
- MTP loop chunked head-seeding merged (O(chunk), identical-token gates).
- Kernel chapter formally closed (ROWS_TG: no dispatch surface).

## Your morning list, in order
1. Read docs/CARD-UPDATE-DRAFT.md (card text) — 10 min.
2. Decide: 397B sidecar to all rungs or just 2.2bpw.
3. Run the push per docs/PUSH-RUNBOOK.md (gated; 2-4h wall, mostly
   unattended — start it before the interview, Flash/35B/27B first).
4. Post LinkedIn (draft has three lengths).
5. Optional 10-min stage-1 smoke: add `export EXO_MTP=1` and
   `export EXO_NO_BATCH=1` to BOTH boxes' ~/.exo/exo-env.sh, restart
   both supervisors (fully — kill the supervisor script, not just exo),
   place Flash 2.1 2-node pipeline, send a greedy request, look for
   "MTP stage 1 engaged" + acceptance ~0.76 in the logs, compare text
   with tonight's scratchpad stage1_single.json. Then remove the env
   lines (or leave them — everything is gated off without them anyway;
   EXO_MTP=1 IS the opt-in).

## Cluster state as left
Both workers up, same code (mtp-stage1 @ 6de2f466), no instances
placed, cache+memory limits armed on both, smoke nodes killed and their
temp homes (~/.exo-smoke*) removable. M4's supervisor is the launchd-
managed one — remember: env/code changes need the SUPERVISOR killed,
not just exo (tonight's stale-supervisor lesson, in the GLM ledger).

Good luck at Apple. The story of the last 24h — measured claims,
refuted theories in the ledger, a gate that catches what testing
missed — is exactly the story worth telling in that room.
