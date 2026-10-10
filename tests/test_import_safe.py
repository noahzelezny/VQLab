"""Every vqlab module must be importable without side effects.

``vqlab/api.py`` calls tools as functions, so ``import vqlab.<stage>.<tool>``
must not parse ``sys.argv``, print, exit or write files. Each module is
imported in a fresh subprocess with a bogus argv and an empty temp cwd;
the import must succeed silently and leave the cwd empty.

``vqlab.runtime`` is excluded (it is the shipped runtime, owned elsewhere).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
PKG = SRC / "vqlab"

# Modules that legitimately act on import, each with its reason.
EXEMPT: dict[str, str] = {
    "vqlab.__main__": "package entry point: `python -m vqlab` runs the CLI by design",
}


def _modules() -> list[str]:
    out = []
    for p in sorted(PKG.rglob("*.py")):
        rel = p.relative_to(SRC).with_suffix("")
        parts = list(rel.parts)
        if parts[1:2] == ["runtime"]:
            continue
        if parts[-1] == "__init__":
            parts = parts[:-1]
        name = ".".join(parts)
        if name in EXEMPT:
            continue
        out.append(name)
    return out


@pytest.mark.parametrize("mod", _modules())
def test_import_is_side_effect_free(mod, tmp_path):
    code = (
        "import sys, importlib\n"
        "sys.argv = ['x', '--definitely-not-a-flag']\n"
        f"importlib.import_module({mod!r})\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=120,
    )
    if r.returncode != 0 and "ModuleNotFoundError" in r.stderr and "vqlab" not in r.stderr.splitlines()[-1]:
        pytest.skip(f"optional dependency missing: {r.stderr.splitlines()[-1]}")
    assert r.returncode == 0, f"import exited {r.returncode}:\n{r.stderr[-2000:]}"
    assert r.stdout == "", f"import printed to stdout:\n{r.stdout[:2000]}"
    assert r.stderr == "", f"import printed to stderr:\n{r.stderr[:2000]}"
    assert list(tmp_path.iterdir()) == [], f"import wrote files: {list(tmp_path.iterdir())}"
