"""cli: the top-level flags and how a tool is run."""
import subprocess
import sys

import pytest

import vqlab
from vqlab import cli


@pytest.mark.parametrize("flag", ["--version", "-V"])
def test_version_prints_the_package_version(flag, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["vqlab", flag])
    assert cli.main() == 0
    assert capsys.readouterr().out.strip() == f"vqlab {vqlab.__version__}"


@pytest.mark.parametrize("cmd", ["kl-ladder", "provenance", "runs"])
def test_a_tool_names_itself_and_runs_without_warnings(cmd):
    # 2026-10-19 review: help said `usage: kl_ladder.py`, and provenance/runs
    # (already imported by the run log) printed a runpy RuntimeWarning
    p = subprocess.run([sys.executable, "-m", "vqlab.cli", cmd, "--help"],
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == 0
    assert p.stdout.startswith(f"usage: vqlab {cmd} ")
    assert "RuntimeWarning" not in p.stderr
