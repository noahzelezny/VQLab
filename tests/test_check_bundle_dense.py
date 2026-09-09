"""Pin that the dense bundle gate separates a MISSING runtime from a DRIFTED one.

The regression this guards: the dense branch decided "bundle is missing
vq_switch.py -- raises ModuleNotFoundError on a stock install" from verbatim
text containment. Containment answers "is this the CURRENT text", not "is the
runtime here at all", so on 2026-09-09 it reported that fatal diagnosis for
all three 27B rungs and gemma-e4b-PLE -- every one of which carries vq_switch
inline (VQSwitchLinear, _dense_fused and _fused are all DEFINED in model.py)
and differs from the repo only by kernel work spliced in after they were
built. They load fine on a stock install; they are merely stale, exactly like
the other 15 artifacts. Four "fatal" artifacts became zero.

The load-bearing case is the FIRST test: a bundle that genuinely lacks the
symbols must still be called out as the stock-install break, or the fix has
traded a false alarm for a silent one.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

GATE = Path(__file__).resolve().parents[1] / "src" / "vqlab" / "check_bundle.py"
RUNTIME_DIR = GATE.parent


def _artifact(tmp_path: Path, model_py: str) -> Path:
    art = tmp_path / "artifact"
    art.mkdir()
    (art / "config.json").write_text(json.dumps({"vq_linear": True}))
    (art / "model.py").write_text(model_py)
    return art


def _run(art: Path):
    return subprocess.run([sys.executable, str(GATE), "--artifact", str(art)],
                          capture_output=True, text=True)


@pytest.fixture(scope="module")
def runtimes():
    return {n: (RUNTIME_DIR / n).read_text()
            for n in ("vq_dense.py", "vq_switch.py")}


def test_genuinely_missing_switch_is_still_the_stock_install_break(tmp_path, runtimes):
    """vq_dense ALONE -> the fused path really would import from site-packages."""
    r = _run(_artifact(tmp_path, runtimes["vq_dense.py"]))
    assert r.returncode == 1
    assert "ModuleNotFoundError" in r.stdout
    assert "VQSwitchLinear" in r.stdout  # names which anchor is undefined


def test_inlined_but_stale_switch_is_reported_as_drift(tmp_path, runtimes):
    """The 27B/gemma shape: every anchor defined, text one kernel behind."""
    stale = runtimes["vq_switch.py"].replace(
        "def _dense_fused", "def _gemmseg_added_after_this_bundle(x):\n    return x\n\n\ndef _dense_fused", 1)
    assert stale != runtimes["vq_switch.py"]
    r = _run(_artifact(tmp_path, runtimes["vq_dense.py"] + "\n" + stale))
    assert r.returncode == 1
    assert "drift, not a missing runtime" in r.stdout
    assert "ModuleNotFoundError" not in r.stdout


def test_a_new_repo_symbol_absent_from_an_old_bundle_is_not_fatal(tmp_path, runtimes):
    """A kernel added to the repo AFTER splicing must not read as absence.

    This is the exact false alarm: gemmseg_cb_dev landed 2026-09-08 and is
    missing from every bundle built before it.
    """
    old_bundle = runtimes["vq_dense.py"] + "\n" + runtimes["vq_switch.py"]
    newer_repo_symbol = "\n\ndef gemmseg_cb_dev_style_addition():\n    return None\n"
    assert newer_repo_symbol not in old_bundle
    r = _run(_artifact(tmp_path, old_bundle))
    assert r.returncode == 0
    assert r.stdout.startswith("PASS")


def test_verbatim_bundle_passes(tmp_path, runtimes):
    r = _run(_artifact(tmp_path, runtimes["vq_dense.py"] + "\n" + runtimes["vq_switch.py"]))
    assert r.returncode == 0, r.stdout
    assert "carries both runtimes verbatim" in r.stdout
