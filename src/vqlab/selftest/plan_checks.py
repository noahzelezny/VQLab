"""Selftest: planning tools (zero-groups, the pricer) and the paper tables."""
from __future__ import annotations

import json
import subprocess

from vqlab._layout import find as _find
from vqlab.selftest import PY, check, run, skip


def check_zero_groups(ctx):
    print("[6a37/7] zero-groups (headers + scales only)")
    import numpy as _np
    from safetensors.numpy import save_file as _sf
    from vqlab.plan import zero_groups as _zg
    _zd = ctx.tmp / "zg"
    _zd.mkdir()
    _m = "model.layers.0.mlp.switch_mlp.up_proj"
    _sc = _np.ones((2, 4, 8), _np.float16)
    _sc[0, :2, :] = 0
    _sc[1, 0, :3] = _np.float16(1e-6)
    _sf({_m + ".codebook": _np.zeros((256, 4), _np.float16),
         _m + ".codes": _np.zeros((2, 4, 8 * 16), _np.uint8), _m + ".vq_scales": _sc},
        str(_zd / "model.safetensors"))
    (_zd / "config.json").write_text(json.dumps({"vq_modules": {_m: {
        "experts": 2, "out": 4, "in": 512, "k": 256, "dim": 4, "group": 64, "pack_bits": 8}}}))
    _zr = _zg.scan_artifact(_zd)["total"]
    check("zero-groups counts exact-zero and sub-normal scales and the code bytes they hold",
          (_zr["groups"], _zr["zero"], _zr["tiny"]) == (64, 16, 19)
          and _zr["zero_bytes"] == 16 * 16 and _zr["tiny_bytes"] == 19 * 16, str(_zr))


def check_pricer(ctx):
    print("[7/7] pricer")
    p = run([str(_find("price.py")), "--family", "qwen397b",
             "--budget-gib", "108"], verbose=ctx.verbose)
    check("price emits a recipe for a byte budget",
          p.returncode == 0 and "harvest recipe" in p.stdout)


def check_paper_tables(ctx):
    # Paper v5: every table regenerates from the saved per-position arrays.
    # The GPU half (the scorer still reproduces them) is research/paper/regress.py.
    from vqlab import _layout as L
    from vqlab import config as _cfg
    paper = L.SRC.parent / "research" / "paper"
    if not (_cfg.scratch() / "paper_rev").exists():
        skip("paper v5 tables regenerate", "per-position arrays not on this machine")
    else:
        p = subprocess.run([PY, "make_results.py"], cwd=paper, capture_output=True,
                           text=True)
        check("paper v5 tables regenerate byte-identical from the saved arrays",
              p.returncode == 0 and p.stdout == (paper / "RESULTS-V5.md").read_text(),
              p.stderr.strip().splitlines()[-1] if p.returncode else "")
