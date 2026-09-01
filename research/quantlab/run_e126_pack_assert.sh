#!/bin/sh
# Independent assertion on E126's pack, per the M4's warning. d2/K512 is the
# most packing-dependent rung either box has built: 15.938 GiB of uint16 codes
# drop to 8.965 at 9 bits, so the artifact is 21.565 GiB UNPACKED and only
# enters Noah's band (<=14.80) if pack_dense actually ran. A silent skip or a
# failed pack leaves a 21.5 GiB artifact that is LARGER than the rungs it is
# meant to beat — and the handoff chain scores "whichever exists", so it would
# report a real number for the wrong artifact.
# Written as a SEPARATE script because the handoff chain is mid-run and a
# running script is never edited.
set -u
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
E="/Volumes/Thunderbay SSD/Exo Models"
L=logs_live_e126_k512.log
while pgrep -f run_e126_k512_handoff >/dev/null; do sleep 30; done
./venv/bin/python - <<'PY' 2>&1 | tee -a $L
import pathlib
E = pathlib.Path("/Volumes/Thunderbay SSD/Exo Models")
G = 2**30
unp = E / "e119-27b-dense-d2k512"
pk  = E / "e119-27b-dense-d2k512-packed"
def sz(p):
    return sum(f.stat().st_size for f in p.glob("*.safetensors")) / G if p.is_dir() else None
u, p = sz(unp), sz(pk)
print(f"PACK ASSERT  unpacked {u:.3f} GiB" + (f"   packed {p:.3f} GiB" if p else "   packed MISSING"))
if p is None:
    print("PACK ASSERT FAIL: no packed artifact — the scored number describes a "
          "21.5 GiB artifact, NOT a 14.59 GiB candidate. Do not put it on the ladder.")
    raise SystemExit(2)
if p > 15.5:
    print(f"PACK ASSERT FAIL: packed size {p:.3f} GiB is not near the predicted "
          f"14.590 — the pack did not do its work. Do not put it on the ladder.")
    raise SystemExit(2)
print(f"PACK ASSERT OK: {u:.3f} -> {p:.3f} GiB ({u-p:.3f} GiB removed), "
      f"predicted 14.590, band <= 14.80")
PY
