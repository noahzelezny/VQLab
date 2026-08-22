# HANDOFF — quantlab, 2026-08-22 ~11:52

Written for a successor session. Everything here is verified at the timestamp
above. **Verify before you act on any of it** — that is the lesson this
session paid for most often.

## HARD DEADLINE: Monday. Noah has said so explicitly.

---

## 1. WHAT IS RUNNING RIGHT NOW

| pid | job | box | notes |
|---|---|---|---|
| 80135 | `fit_dense_vq` d2/K4096 (E128 run C) | M3 | R3 slope test, L31 of 63 at 8342s, ~113 s/tensor |
| 81035 | `run_e130_rate_twin.sh` | M3 | ARMED, waits for run C. d2/K64 vs d4/K4096, both 11.60 GiB |
| 81414 | `hf upload-large-folder` | M3 | **101 GiB publish in flight** — see §2 |

On the M4 (peer session, see §6): R2 = `e128-35b-d4K16384` fitting, then the
**vintage test** queued behind it.

**Strictly sequential is the rule.** Concurrency killed E120 and E121 last
night and invalidated a timing measurement this morning. Do not run two GPU
jobs on one box.

## 2. THE PUBLISH IN FLIGHT — FINISH THIS FIRST

Replacing the published `TheDrainFlorist/Qwen3.5-397B-A17B-VQ-2.2bpw` with
`rotlab--397B-d8K16384-packed` (d8/K16384). Noah authorized it explicitly.
Same repo deliberately, to preserve download metrics. Zero orphan files
(checked: all 38 remote files are overwritten).

**Remaining steps, in order:**
1. Wait for the upload to finish. Do NOT trust the progress bars — verify by
   diffing `HfApi().list_repo_files(...)` against the local artifact.
2. **Push `README.md` explicitly** afterwards. Batch ordering means it may not
   land otherwise; the card is the last thing to verify, not the first.
3. Fetch the rendered card back and read it. Confirm the v2 notice, the
   `revision=` pin, and that all 9 sections survived.

**The card is an UPDATE of the published card, not a rewrite.** Noah was
explicit. I nearly shipped a from-scratch card built against a STALE LOCAL
copy of the README (143 lines) instead of the live published one (214) — it
would have deleted six sections including the task-benchmark table. The
current file is the published card plus 5 hunks. Keep it that way.

Also fixed a pre-existing live error: the card claimed "~25 GiB free on a
128 GB box"; the real figure is ~17.4 GiB (101.8 GiB resident at 8k against
119.2 GiB usable). Same GB/GiB confusion appears repeatedly — **a "128 GB"
machine is 119.2 GiB. 101 GiB = 108.4 GB. Say which unit and stick to it.**

**Task benchmarks were NOT re-run** (Noah: "I don't want to rerun it. Maybe
later"). The table's first row is relabelled `(v1 weights)` with a note that
v2 has not been re-evaluated. Do not quietly reuse v1's task numbers for v2.
If it is ever wanted: ~90 min, M3 only (lm-eval exists only there; a fresh
install elsewhere breaks the paired McNemar comparison), `LIMIT=1000
./run_task_bench.sh` after adding the d8 dir to MODELS.

## 3. THE LADDER (E128) — Noah's three offerings per model

    R1 accessibility — exceed 4-bit quality at REDUCED size
    R2 daily        — match affine 4-bit SIZE at meaningfully better quality
    R3 heavy weight — affine 8-BIT+ quality at meaningfully smaller size

    27B  (q4 14.094 GiB / KL 45.842 / ppl 5.2055 ; q8 26.341 / KL 1.641)
      R1  E124 d2/K256   13.596 GiB  KL 40.327   MET
      R2  E126 d2/K512   14.592 GiB  KL 33.095   MET (size +3.5% over q4)
      R3  run C in flight — d2/K4096, 17.58 GiB projected
    35B  (q4 19.0 GiB / KL 78.557 ; q8 35.131 / KL 7.449)
      R1  e94b d4/K8192  17.651 GiB  KL 53.022   MET
      R2  M4 fitting d4/K16384, 18.59 GiB projected
      R3  UNPICKED — peer holding until run C reports

**R3 may be unreachable on both models.** The 8-bit bars are 20x and 7x
beyond our best. Fitted d2 slope is x0.673 KL per +1 bpw; even 7 bpw
extrapolates to ~12 mnats against a 1.641 bar. Run C is the decisive test —
if it lands near ~18 mnats, say plainly that R3 is out of reach rather than
burning the weekend on it.

## 4. OPEN DECISIONS — NOAH'S, NOT YOURS

1. **K65536** (the exact rate twin of R1, d4, 13.594 GiB). Measured cost
   ~40-60 h, not 4. Noah called the weekend the window but never gave a go,
   and was "dubious of the estimate" — rightly, every timing figure was taken
   under contention. E130 may make it moot.
2. **`chmod -R a-w` on scored artifacts.** Two silent in-place overwrites
   happened (E94's scored artifact; the base's mtimes). `artifact_manifest.py`
   now stamps and checks; the chmod half is Noah's call and must not be done
   while fits are writing.
3. **A second dense family** — the 27B result is one model. Not a task you
   can run; Noah must acquire one or the paper states the limitation.
4. **Publishing anything else.** Only the 101 GiB swap was authorized.

## 5. THE VINTAGE GAP — STILL UNEXPLAINED, AND NOAH CARES

Nothing we build reproduces the shipped 2.4bpw's 2.7655 wikitext.

    shipped 2.4        2.7655 / 2.6383    (fit on the M4, Aug 15-16)
    refit ++  (E92)    2.8057 / 2.6447    (M3)
    randinit  (E117)   2.8158 / 2.6347    (M3)  <- BEST on code
    fitter0816 (E121)  2.8292 / 2.6508    (M3)  <- the actual 08-16 code

**Falsified:** seeding (E117), a K-crossover (E118), summation order (E120 —
real but ~2.4e-6, far too small), the fitter file (E121), and provenance
(retracted 5ab7f7a — 2032/2032 non-VQ tensors and 171/171 vq_scales are
byte-identical; an mtime is not an input).

**What survives:** the shipped fit ran on the **M4**; every refit ran on the
**M3**. The box has never been an arm. The gap is **wikitext-only** — on code
the shipped artifact is SECOND. Its relerr is ordinary (0.3156 vs randinit's
0.3157). An n=2 397B floor already exists at 0.0134 wikitext against a
0.040 gap.

**The queued M4 fit tests this.** Readings were AMENDED before it runs
(0d9fbd9): the M4's mlx was reinstalled Aug 17, AFTER the shipped fit, so a
null excludes "the box as it is today", NOT "the box". H4 — the mlx version
the M4 carried on Aug 15 — is unrecovered; searched repo pins, all git
branches, shell history, both pip caches, `~/mlx-wheels` and
`~/mlx-jaccl-fork` (both June-dated).

Noah's standing view, and he is right: *"until you can reproduce it, it's an
empty claim... this feels less like research and more like luck."*

## 6. PEER SESSIONS

- **M4 / task-suite** (`agenicai-16`, socket 39597) — owns the M4, the
  lm-eval harness, R2, and the vintage fit. Excellent: caught the E101
  confound, the unpacked-size error, the mlx-reinstall asymmetry, and the
  pack_dense deferred read. Treat its corrections as usually right and verify
  anyway — it has also been wrong (the "ppl came off unpacked bytes" claim).
- **Paper session** (`quantlab-56`) — cites ONLY committed EXPERIMENTS.md
  entries. Ranked what the paper needs: (1) a clean law-6 specimen [done,
  E127], (2) a second dense family [blocked, Noah's], (3) the rate twin
  [E130, running].
- **Web GUI** (`quantlab-88`) — renders the ladder; keeps rebuildability as a
  first-class (family, d, K) field.

## 7. HOW TO NOT REPEAT THIS SESSION'S MISTAKES

Each of these cost real time or nearly shipped a defect:

- **Check before you speak.** Most errors here were stating something from
  memory that a 30-second command would have corrected: a rule that did not
  exist (III.10/III.11), a sweep that found 4 files when it was 11, a smoke
  that "hadn't run" when E103 had run it, a card that "had no benchmarks."
- **An mtime is not an input.** Compare CONTENT.
- **`preflight_ram.py` before ANY resident-memory op.** Three violations in
  one day; that is why the guard exists.
- **Small-case first.** A 1-expert or 1-layer run costs seconds and caught two
  real bugs.
- **Never edit a running chain's script.** Kill the watcher, re-arm.
- **Pre-register readings, including the inconclusive branch**, before a run.
- **Report ratios, not absolutes**, for anything speed-related — the decode
  meter is bimodal and unexplained.
- **Seed-noise floor (6f): KL 2.085 mnats / ppl 0.0447 at 27B d2/K256, n=3.**
  Third-decimal ppl differences between single-draw artifacts are NOT
  interpretable. This retracted one claim and validated another the same night.

## 8. DOCS

`EXPERIMENTS.md` through E130 · `FINDINGS.md` (laws, incl. 6b-6f, III.1-III.11a)
· `PROCESS.md` (family onboarding, preflight, provenance) · `STATE.md` ·
`manifests/` (11 artifacts stamped).

`MODEL_CARD_397B_G.md` has uncommitted edits that are NOT mine — leave them.
