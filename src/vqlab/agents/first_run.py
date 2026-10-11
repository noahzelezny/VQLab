"""First run: give a new install its standard locations.

`pip install` cannot run anything, so the first `vqlab` command does it:
when there is no config file, no location set in the environment and no
data from an older install, it creates the lab folder (``~/VQLab``) with
``models/``, ``scratch/``, ``fits/`` and ``records/`` (bf16 teachers share
``models/`` until set apart),
writes a config file naming it, and says so once. Moving it later is
`vqlab config set lab PATH` or the GUI's Storage page.

``$VQLAB_NO_SETUP=1`` skips it (the test suite sets it).
"""
from __future__ import annotations

import os
import pathlib
import sys

from vqlab import config as C
from vqlab.agents import settings as S

SUBFOLDERS = ("models", "scratch", "fits", "records")   # teachers default to models/
LOW_SYSTEM_FREE = 500 * 2**30      # teachers are 50-750 GiB each


def needed() -> bool:
    if os.environ.get("VQLAB_NO_SETUP"):
        return False
    if C.config_file().exists():
        return False
    if any(os.environ.get(v) for v in C._ENV.values()):
        return False
    return not C._legacy_home_used()


def run(out=sys.stderr) -> pathlib.Path | None:
    """Create the lab folder and config if this is a first run; returns the
    lab folder when it did, else None. Never raises: a failure only means the
    defaults stay unconfigured, so it is reported and the command goes on."""
    if not needed():
        return None
    lab = C.lab()
    try:
        for d in SUBFOLDERS:
            (lab / d).mkdir(parents=True, exist_ok=True)
        cf = C.config_file()
        cf.parent.mkdir(parents=True, exist_ok=True)
        cf.write_text("# vqlab storage (vqlab config / the GUI's Storage page edit this file)\n"
                      f'[paths]\nlab = "{lab}"\n')
    except OSError as e:
        print(f"vqlab: could not set up {lab} ({e}); set a location with "
              f"`vqlab config set lab PATH`", file=out)
        return None
    S._reset()
    free = S.free_bytes(lab) or 0
    print(f"vqlab: first run. Models, teachers and work in progress go in {lab}\n"
          f"       ({free / 2**30:,.0f} GiB free there). Config: {C.config_file()}\n"
          f"       Move it any time: `vqlab config set lab PATH`, or `vqlab gui` -> Storage.",
          file=out)
    try:                                   # the models already on this machine
        from vqlab.agents import model_scan
        rows = model_scan.refresh()
        n = sum(1 for r in rows if r.get("use") in ("original", "quantized"))
        if n:
            print(f"       Found {n} model(s) already on this machine (`vqlab lab scan`).", file=out)
    except Exception:
        pass
    if free < LOW_SYSTEM_FREE and lab.stat().st_dev == pathlib.Path.home().stat().st_dev:
        print("       This is the system disk with under 500 GiB free; teacher models are "
              "50-750 GiB each. Consider a larger drive.", file=out)
    return lab
