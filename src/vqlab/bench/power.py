"""GPU package power, read from macmon (the box_quiet / timing benches'
measure of whether the GPU is actually doing work)."""
from __future__ import annotations

import json
import shutil
import subprocess


def gpu_watts() -> float:
    """Instantaneous GPU package watts, or -1 if macmon is unavailable.

    Utilization is NOT usable here: WindowServer and WebKit peg gpu_usage to
    60-95% while drawing 2-4 W, so a utilization gate both never opens and
    reads 'busy' on an idle box. Power separates compositing from compute.
    """
    try:
        out = subprocess.run(
            [shutil.which("macmon") or "macmon", "pipe", "--interval", "700", "-s", "1"],
            capture_output=True, text=True, timeout=6).stdout.splitlines()
        return float(json.loads(out[0])["gpu_power"])
    except Exception:
        return -1.0
