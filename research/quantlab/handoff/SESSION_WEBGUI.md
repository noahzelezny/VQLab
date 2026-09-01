# Session brief: the web GUI

Repo: /Users/noahzelezny/Documents/AgenicAI/quantlab. Work ONLY in webgui/.
Do not touch anything else; another session is running live experiments.
No GPU work. No servers left running.

A partial prototype may exist in webgui/ — inspect it first, keep what
works, and iterate. Zero build step, zero network deps (no CDNs), one
index.html plus optional local .js/.css, works from file://. Dark + light.
Aesthetic reference: exo's localhost dashboard — clean, flat, minimal.

CENTREPIECE: the size-targeting slider. Drag a target size in GiB; get the
recommended build (nearest flat node, or node + harvest depth), predicted
size, estimated quality, fit-time estimate, and d/K feasibility.

Measured constants (from EXPERIMENTS.md/FINDINGS.md — read for context):
- 397B ladder, post-graft GiB / prose ppl, one instrument:
  flat K128 100.9/3.1706; flat K256 refit ~111.6/PENDING; flat K512
  ~122.3/PENDING; harvest 139.93/2.3452; flat K2048 refit 143.65/2.3410.
- vision tower = 0.849 GiB exactly. Size model: shallow 1.87 GiB/bit,
  body 8.81 GiB/bit (6-for-6 out-of-sample).
- harvest cost: ~0.0011 ppl/GiB at K2048 base, ~0.0033 at K256, 0.024-0.032
  at K128.
- geometry: bits/weight = log2(K)/d; d in {2,4,8,16}; uint8 K<=256, uint16
  above; codebook bytes = K*d*2; 32KB threadgroup fast-kernel limit;
  K > 2M samples = degenerate.
- comparators (grey): spicyneuron 2.6bit 121.0/3.1843, 3.5bit 165.6/2.3614.

HONESTY IS A UI REQUIREMENT: measured = solid dots, estimates = hollow,
today's pending verdicts = "measuring today" badge. Never render an
estimate as if measured.

Screenshot-test and iterate before showing Noah.
