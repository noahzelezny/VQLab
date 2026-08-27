# Experiment timeline

Every experiment in this project, in order, with what each one settled.
Extracted verbatim from `EXPERIMENTS.md` headings by `make_timeline.py` --
no paraphrase, so nothing here can drift from the log it came from.

**This is the receipt.** The model artifacts are large and most are
deletable; the finding, its date, and its line in the log are not. For any
row, `EXPERIMENTS.md:<line>` has the full entry with metric values,
instrument, and conditions.

**73 experiments, 24 with a written result, 19 pre-registered, 3 pre-registered and never closed.**

Dates marked `~` are inferred from position in the log, not recorded in
the entry, and can be off by days. The ORDER is exact regardless; where a
date matters, `paper/LEDGER.md` carries dated entries for the published
results.

`pre-reg` marks a prediction registered BEFORE the numbers existed. A
pre-reg with no matching result is shown as such rather than hidden --
those are the questions this project asked and did not answer.

| # | date | kind | what it settled | log |
|---|---|---|---|---|
| E1 | 08-08 | entry | Vision audit. spicyneuron 2.6/3.5bit 397B = text-only (mlx-lm convert strips vision); mlx-community 4bit (mlx-vlm convert) keeps it. Root cause of ... | [123](EXPERIMENTS.md#L123) |
| E2 | 08-08 | entry | exo vision pipeline debugging — 4 stacked bugs fixed (card capabilities, Scout probe cache, sidecar-blind loader glob, unpatched 2nd node). OptiQ-9... | [126](EXPERIMENTS.md#L126) |
| E3 | 08-08/09 | entry | 9B PPL baseline + resume test. 8bit 8.70 / OptiQ~6bpw 8.95 / flat-4bit 9.24. Resume deterministic. [e47cd33e, c91de858] | [129](EXPERIMENTS.md#L129) |
| E4 | 08-09 | entry | Matched-budget dense shootout: calibrated 2.72bpw 19.80 beats static 2.77-2.78bpw 29.1-29.7. Static targeting unreliable at low bpw. [c91de858] | [131](EXPERIMENTS.md#L131) |
| E5 | 08-09 | entry | 35B MoE ladder: calibrated ships 18.3G claiming 2.6bpw and still loses to 11.3G static. (Absolute PPLs from this M4 run later invalidated by the ha... | [133](EXPERIMENTS.md#L133) |
| E6 | 08-09 | entry | Hybrid (attn floor + budget fix): honest 2.61bpw allocation, but at 13.2G / PPL 8.75 sits above the static frontier (t2.6: 12.9G / 8.30). Fixed cal... | [136](EXPERIMENTS.md#L136) |
| E7 | 08-09 | entry | Falsification: forced attn-only-2bit (46.37, 18.3G) vs experts-only-2bit (10.37, 10.9G), trusted harness. Isolation assumption dead. | [139](EXPERIMENTS.md#L139) |
| E8 | 08-09 | entry | Complete 35B ladder on the trusted harness (single scale, final): | [141](EXPERIMENTS.md#L141) |
| E9 | 08-09 | entry | Operator-designed allocation (attn=6, experts=2, other=4) via forced mode: 11.3G / 10.19. Beats the crude crush (10.9G / 10.37) — attention enrichm... | [164](EXPERIMENTS.md#L164) |
| E10 | 08-10 | entry | Complete 511-target × 5-bit sweep, true bf16 reference, M4 (10.5h, die 64-70C on a USB fan, zero throttle events). First fully-honest fine-grained ... | [173](EXPERIMENTS.md#L173) |
| E11 | 08-10 | entry | Depth-law falsification — **REFUTED.** Two matched builds (`--method static --target-bpw 2.6 --candidate-bits 2,3,4`, same bf16 snapshot, trusted h... | [221](EXPERIMENTS.md#L221) |
| E12 | 08-10 | entry | Depth-shape ladder — completes the E11 axis. Four matched builds (2.601 bpw, ~13.9G, same recipe/harness; `OPTIQ_DEPTH_PRIOR=flat\|reverse` added a... | [248](EXPERIMENTS.md#L248) |
| E13 | 08-10 | entry | Steepness ladder — **degenerate at fixed budget.** k=1.35 and k=5.5 reverse-curves produced allocations bit-identical to k=2.73 (0 of 511 layers di... | [278](EXPERIMENTS.md#L278) |
| E14 | 08-10 | entry | Hybrid prior — **the decomposition validated in the most literal way possible.** At each budget the hybrid built the BIT-IDENTICAL artifact to that... | [307](EXPERIMENTS.md#L307) |
| E15 | 08-11 | entry | First 397B production builds (t2.6 + t2.4, reverse_experts, streaming, M3) — and the scale-up asteroid field: **13 takes, 7 real defects, one envir... | [327](EXPERIMENTS.md#L327) |
| E16 | 08-11 | entry | DWQ-on-MoE dry run — **possible AND helpful.** mlx_lm dwq, 35B bf16 teacher (65G) + optiq mixed revexp-t2.4 student (12G) on the M4: peak 110.9G of... | [368](EXPERIMENTS.md#L368) |
| E94 | 08-21 | result | fitter vintage confirmed at 35B, second family | [5223](EXPERIMENTS.md#L5223) |
| E95 | 08-21 | pre-reg | dense VQ on Qwen3.8-27B (flat, no tail) | [5091](EXPERIMENTS.md#L5091) |
|  |  | result | dense VQ carries. The recipe is NOT an MoE-expert phenomenon. | [5132](EXPERIMENTS.md#L5132) |
| E96 | ~08-11 | entry | falsified: the scatter-add port did not speed up 35B/K8192 | [5059](EXPERIMENTS.md#L5059) |
| E97 | ~08-21 | entry | the corruption was visible in our own recorded numbers, unread | [5354](EXPERIMENTS.md#L5354) |
| E98 | ~08-21 | entry | both K8192 arms PASS; and the pair is the cleanest law-6 specimen we have | [5397](EXPERIMENTS.md#L5397) |
| E99 | ~08-21 | entry | the repaired d2 arm: d4's margin was ~3x inflated by contamination | [5430](EXPERIMENTS.md#L5430) |
| E100 | ~08-21 | entry | the d8 artifact cannot serve PACKED; unpacked it runs but loses | [5461](EXPERIMENTS.md#L5461) |
| E101 | ~08-21 | entry | the K256 refit is NOT corrupt. It fits BETTER and scores WORSE. | [5527](EXPERIMENTS.md#L5527) |
| E102 | ~08-21 | entry | WHY the K256 refit loses: it trades the tail for the bulk | [5566](EXPERIMENTS.md#L5566) |
| E103 | ~08-21 | entry | flagship SERVES on the 2-node exo ring, coherent on graded probes | [5613](EXPERIMENTS.md#L5613) |
| E104 | ~08-21 | entry | every packed artifact declared the UNPACKED size | [5654](EXPERIMENTS.md#L5654) |
| E105 | ~08-21 | entry | tail-weighted k-means: screen PASSES, and plain k-means was never minimizing weight-space MSE | [5686](EXPERIMENTS.md#L5686) |
| E106 | ~08-21 | entry | the tail-weight failure is in the SEEDING, not the update; and the screen was testing half the patch | [5732](EXPERIMENTS.md#L5732) |
| E107 | ~08-21 | entry | k-means++ seeding buys the bulk with the tail, and only pays off when K is large | [5771](EXPERIMENTS.md#L5771) |
| E108 | ~08-21 | entry | 2 of 3 tensors. The mechanism is real but NOT universal. | [5822](EXPERIMENTS.md#L5822) |
| E109 | ~08-21 | entry | the ++ penalty is DEPTH-STRUCTURED, and it is a body-layer effect | [5872](EXPERIMENTS.md#L5872) |
| E110 | ~08-21 | entry | WHY the depth flip: shallow layers are heavy-tailed, body layers are sub-Gaussian | [5916](EXPERIMENTS.md#L5916) |
| E112 | ~08-21 | pre-reg | PRE-REGISTERED READING RULE (written before the scores exist) | [6022](EXPERIMENTS.md#L6022) |
|  |  | result | FALSIFIED. Tail-weighting the body makes the model WORSE. | [6066](EXPERIMENTS.md#L6066) |
| E113 | ~08-21 | entry | packed d=8 fused kernel: CORRECT on the real artifact, speed unmeasured | [5965](EXPERIMENTS.md#L5965) |
| E115 | ~08-22 | entry | d8 packed speed A/B: ~19% decode tax, and a decode instrument that is bimodal at 100 GiB | [7310](EXPERIMENTS.md#L7310) |
| E116 | ~08-22 | entry | the SMB write path is clean on a real-artifact round-trip | [7378](EXPERIMENTS.md#L7378) |
| E117 | ~08-21 | pre-reg | K256 --init random, end-to-end (launched 18:47, M3) | [6118](EXPERIMENTS.md#L6118) |
|  |  | result | FALSIFIED. Init is NOT what separates the shipped 2.4 from the refits. | [7214](EXPERIMENTS.md#L7214) |
| E118 | ~08-21 | pre-reg | K512 --init random (queued behind E117) | [6147](EXPERIMENTS.md#L6147) |
|  |  | result | random init WINS at K512. The K-dependence is NON-MONOTONE. | [6345](EXPERIMENTS.md#L6345) |
| E119 | 08-21 | result | the d4 dense K ladder. K1024 beats q3 at less than q3's size. | [6407](EXPERIMENTS.md#L6407) |
|  |  | pre-reg | the 27B dense K ladder (M4, Noah's directive 08-21 ~20:00) | [7248](EXPERIMENTS.md#L7248) |
| E120 | ~08-22 | pre-reg | the vintage hunt, narrowed to float summation order **(no result)** | [7163](EXPERIMENTS.md#L7163) |
| E121 | 08-16 | pre-reg | run the 08-16 fitter itself (queued, M3) | [6234](EXPERIMENTS.md#L6234) |
|  |  | result | NOT-THE-FILE. (The "VOID" verdict below is itself retracted — see the 08-22 bracket.) | [7031](EXPERIMENTS.md#L7031) |
| E122 | ~08-16 | pre-reg | publish-readiness audit of the two swap candidates | [6268](EXPERIMENTS.md#L6268) |
|  |  | result | the d8 re-score is BIT-IDENTICAL, and that is the finding. | [6370](EXPERIMENTS.md#L6370) |
| E123 | ~08-22 | entry | the zeroed-tensor collapse is a DEFERRED READ returning zeros | [7429](EXPERIMENTS.md#L7429) |
| E124 | ~08-21 | pre-reg | the 27B at 4-BIT SIZE, chasing 8-bit quality | [6158](EXPERIMENTS.md#L6158) |
|  |  | result | beats q4 on KL and top-1, at less than q4's size. Loses on ppl. | [6981](EXPERIMENTS.md#L6981) |
| E125 | ~08-22 | result | the fitter is NOT bitwise reproducible, but IS statistically reproducible. E94's numbers are recovered. | [6933](EXPERIMENTS.md#L6933) |
| E126 | ~08-21 | pre-reg | d2/K512, and the 6c replication test | [6485](EXPERIMENTS.md#L6485) |
|  |  | result | d2/K512 beats q4 on ALL THREE metrics. Misses the pre-registered ppl MARGIN. | [6885](EXPERIMENTS.md#L6885) |
| E127 | ~08-21 | pre-reg | a clean law-6 specimen on the dense 27B | [6517](EXPERIMENTS.md#L6517) |
|  |  | result | TRACKS on ppl. And the floor it measured retracts E126's ppl claim. | [6825](EXPERIMENTS.md#L6825) |
| E128 | 08-22 | entry | three rungs per model (Noah, 08-22, hard deadline Monday) | [6726](EXPERIMENTS.md#L6726) |
| E129 | ~08-21 | investigation | the vintage gap, restated correctly. It may be a seed lottery on one corpus. | [6599](EXPERIMENTS.md#L6599) |
|  |  | result | H2 EXCLUDED. The vintage gap is CLOSED AS UNEXPLAINED. | [7967](EXPERIMENTS.md#L7967) |
| E130 | ~08-22 | pre-reg | the d-vs-K rate twin at 3.00 bpw (armed, M3) | [7517](EXPERIMENTS.md#L7517) |
|  |  | partial | arm 1 only. Arm 2 never started. | [7709](EXPERIMENTS.md#L7709) |
| E131 | ~08-22 | result | the 101 GiB swap is PUBLISHED, and the progress bar lied for 74 minutes | [7551](EXPERIMENTS.md#L7551) |
| E132 | ~08-22 | pre-reg | the M3 scoring queue (arm 2, R2, packed e94b) **(no result)** | [7724](EXPERIMENTS.md#L7724) |
| E133 | ~08-22 | pre-reg | a 27B q6 comparator, to extend claim 1's fence to 6 bpw | [7786](EXPERIMENTS.md#L7786) |
|  |  | result | q6 WINS. The VQ/affine crossover is BRACKETED in 4.5-6.0 bpw. | [8152](EXPERIMENTS.md#L8152) |
| E134 | ~08-22 | pre-reg | the d4 large-K fused kernel cannot load. Mechanism named BEFORE the M3 test. | [7833](EXPERIMENTS.md#L7833) |
|  |  | result | CONFIRMED architectural. Both 35B rungs are UNRELEASABLE on BOTH boxes. | [7897](EXPERIMENTS.md#L7897) |
| E135 | 08-20 | result | the E134 device-codebook kernel is ACCEPTED. And the runtime that executes is not the one an artifact bundles. | [8043](EXPERIMENTS.md#L8043) |
| E136 | ~08-20 | result | 2.7706. THE SHIPPED RECIPE IS RECOVERABLE. The axis is the INTERPRETER STACK. | [8199](EXPERIMENTS.md#L8199) |
| E136b | ~08-20 | result | the replicate WITHDRAWS E136's headline. The interpreter axis is NOT established. | [8461](EXPERIMENTS.md#L8461) |
| E137 | ~08-20 | entry | A BUNDLED RUNTIME CAN BE BOTH EXECUTING AND BYPASSED FOR THE LAYER THAT MATTERS | [8278](EXPERIMENTS.md#L8278) |
| E138 | ~08-20 | pre-reg | 27B d4/K65536, the exact rate twin of E124 (armed, M3) | [8545](EXPERIMENTS.md#L8545) |
|  |  | result | MIXED, and the registered branches did not fire cleanly. d2 wins on ppl; d4's KL edge is inside a distrusted floor. | [8825](EXPERIMENTS.md#L8825) |
| E139 | ~08-20 | entry | THE FITTER IS NOW SEEDED. And seeding is NECESSARY BUT NOT SUFFICIENT. | [8602](EXPERIMENTS.md#L8602) |
| E143 | 08-24 | entry | III.11 ring smoke of rotlab/397B-flatk512-packed (publish gate) | [9198](EXPERIMENTS.md#L9198) |
| E144 | 08-24 | entry | rebuild of the 27B q8 comparator (RESULT CONTRADICTS PRE-REGISTRATION) | [9233](EXPERIMENTS.md#L9233) |
| E145 | 08-24 | entry | full config audit of every rung (79 dirs). One new asymmetry, and it favors US. | [9277](EXPERIMENTS.md#L9277) |
| E146 | 08-24 | entry | the spicyneuron comparators are MIXED-ALLOCATION, and text-only | [9348](EXPERIMENTS.md#L9348) |
| E147 | 08-24 | entry | long-prompt prefill, e4b VQ-PLE vs the 8-bit incumbent | [9423](EXPERIMENTS.md#L9423) |
| E136-M4 | ~08-20 | entry | the Aug-15 stack, reconstructed and REPLICATED (n=2, M4) | [8371](EXPERIMENTS.md#L8371) |
| E140-M3 | ~08-20 | retraction | "the codes cluster by BOX" is NOT supported. Raised by the M4 session against its own claim. | [8685](EXPERIMENTS.md#L8685) |
| E140-M4 | ~08-20 | result | 35B d2/K1024 (result lives in paper/LEDGER.md, not here) | [8738](EXPERIMENTS.md#L8738) |
| E141-M3 | ~08-20 | pre-reg | did the thin init starve E138? (L34, high-impact body layer) | [8882](EXPERIMENTS.md#L8882) |
|  |  | result | STARVATION RULED OUT. The 200k init cap does not depress K=65536. | [8926](EXPERIMENTS.md#L8926) |
| E141-M4 | ~08-20 | pre-reg | 35B d2/K4096, the matched-byte head-to-head with q6 **(no result)** | [8767](EXPERIMENTS.md#L8767) |
| E142-27B | ~08-20 | pre-reg | does E127's iters lever transfer to d2/K512? (two arms, M3, seed 1234) | [8974](EXPERIMENTS.md#L8974) |
|  |  | result | the iters lever does NOT transfer to K512. Arm 2 adopted anyway, on stated grounds. | [9064](EXPERIMENTS.md#L9064) |
| E142-397B | ~08-20 | result | the d4/K2048 fit-to-fit floor. 0.0056 prose / 0.0104 code. | [9121](EXPERIMENTS.md#L9121) |
