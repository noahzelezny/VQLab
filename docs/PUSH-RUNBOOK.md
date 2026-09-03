# Morning push runbook — 2026-09-03

Everything below is gated: `vqlab publish` refuses on any gate failure,
re-hashes files around the gate, and now also refuses to certify from a
patched venv. Nothing here has been run; nothing uploads without you.

## Preconditions (all DONE overnight)

- 18/18 artifacts re-bundled with the final kernel (devx+simd_sum),
  `.pre-arc5` backups beside each; static gates PASS on all;
  full check-release (incl. smoke) PASS on Flash 2.1bpw + the 3x 27B.
- Referee on re-bundled 2.1bpw, chunk-16: identical to 16 digits.
- Sidecars staged: Flash x4 (2.14G each), GLM 2.7 (6.09G), 397B 2.2 (5.41G).
- GLM memory story closed (cache-limit; ledger 9c99567).
- NO-SHIP: gemma-4-e4b-it-VQ-PLE (pre-existing Metal threadgroup defect,
  ledger 0e4d5ec). It is excluded below.

## Your two review items before any command runs

1. Card text: docs/CARD-UPDATE-DRAFT.md (the [ALL]/[FLASH]/[397B]
   sections become per-repo README edits).
2. Decide 397B sidecar placement: it's staged in the 2.2bpw dir only
   (the rung that fits one 128G box). Copy to other 397B rungs or not.

## Environment

```bash
export HF_HOME="/Volumes/Thunderbay SSD/Mlx_Models"
cd ~/Documents/AgenicAI/vqlab
```

## Commands (per repo; run a --dry-run of the first one as a canary)

Runtime + card update (most repos):

```bash
python -m vqlab.cli publish --artifact "/Volumes/Thunderbay SSD/Exo Models/TheDrainFlorist--Qwen3.8-Flash-Next-VQ-2.1bpw" --repo TheDrainFlorist/Qwen3.8-Flash-Next-VQ-2.1bpw --files model.py README.md mtp-head-q6.safetensors --message "2026-09 refresh: faster kernels (1-ULP equiv, zero quality delta), MTP sidecar, serving guide"
```

- MTP repos (Flash x4, GLM-2.7, 397B-2.2): include `mtp-head-q6.safetensors` in --files.
- Non-MTP repos: `--files model.py README.md` only.
- The 27B x3: same, plus `vq_switch.py` if check-bundle placed it as a
  separate file (verify with `ls` first — the dense re-bundle may have
  inlined it; whatever is in the artifact dir now is what passed the gate).
- The gate loads and generates per repo (~minutes each for the big ones).
  Total wall estimate: 2-4 h for the full set; Flash/27B/35B first
  (fast), 397B/GLM last (big loads).

Full artifact list (17): Flash-Next 2.1/3.2/4.4/5.5 - 397B 2.2/2.4/2.6/3.1
- GLM 2.7/3.1/3.6 - 35B 3.4/3.8/4.6/5.4 - 27B 3.9/4.5/4.8 - gemma-26b 6.2.
(That's 18 minus gemma-e4b-PLE. Count check: 4+4+3+4+3+1 = 19 — the 26b
is the 19th; PLE excluded from 20 total backups... verify against
`ls "/Volumes/Thunderbay SSD/Exo Models" | grep TheDrainFlorist` before
starting and reconcile with the .pre-arc5 markers.)

## After the push

- Update docs/CARD-UPDATE-DRAFT.md checklist: mark pushed + note the new
  revision hashes as pinnable.
- LinkedIn: docs/LINKEDIN-DRAFT.md (post after links resolve).
- exo fork: `mtp-stage1` branch holds tonight's serving fixes (cache
  limit, DownloadFailed retry, stage-1 MTP). Merge to main + push the
  fork when you're ready; the runbook deliberately leaves fork-publishing
  as its own decision.
