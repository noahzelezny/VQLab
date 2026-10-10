"""Selftest: the vq-skipzero experiment (pack -> expand on CPU, and the
CPU half of the sz-bitexact gate in its own process)."""
from __future__ import annotations

import contextlib
import io
import os
import subprocess
import sys

from vqlab import _layout
from vqlab.selftest import check


def check_skipzero(ctx):
    print("[6a39/7] vq-skipzero experiment (pack -> expand, CPU)")
    from vqlab.selftest import sz_check_fixture
    with contextlib.redirect_stdout(io.StringIO()):
        _szok = sz_check_fixture.run(root=ctx.tmp / "skipzero")
    check("vq-skipzero: live rows byte-identical after expansion, dead rows exactly zero",
          _szok is not False)

    # its own process: the checks set the default device and exec kernel text
    _szr = subprocess.run([sys.executable, "-m", "vqlab.selftest.sz_bitexact_cpu"],
                          capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(_layout.SRC)})
    check("vq-skipzero: runtime carries the SZ switch natively; the experimental forks still "
          "apply to the frozen walk/u8 vintages; row table + resident weights (CPU)",
          _szr.returncode == 0
          and "carries the SZ switch natively: OK" in _szr.stdout
          and "forks apply to walk runtime" in _szr.stdout,
          (_szr.stdout + _szr.stderr).strip().splitlines()[-1] if _szr.returncode else "")
