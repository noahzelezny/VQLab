"""The vision surface check must be IN the gate everyone runs, and must fail.

F153 (2026-09-19): 17 of 20 shipped multimodal bundles carried a vision tower
their runtime could not reach, for three weeks, with a green board the whole
time. Every gate passed honestly — `check-release` found every file present,
`check-bundle` found the right runtime text, `smoke` generated a token —
because a text-only arch serves text perfectly well. The gate set simply never
asked whether the shipped vision weights were reachable.

So the failure was not a missing check, it was a missing QUESTION. Two things
have to stay true or it comes back:

  1. The vision surface check is wired into `vqlab check`, the cheap composite
     that runs before release — not only available as a command someone has to
     remember. A gate nobody runs is a gate that does not exist.
  2. It actually FAILS on the historical bug. A gate that passes everything is
     indistinguishable from no gate, which is precisely how this shipped: the
     old checks were all green on all 17 broken artifacts.

Test 2 is a mutation test, not a mock. It reproduces the exact pre-fix rule —
arch resolution ordered mlx_lm-first regardless of modality — against a real
bundle, and demands a non-zero exit. Verified when written: PASS on the fixed
gemma-4-e4b-it-VQ-PLE, FAIL with the F153 diagnostic on the mutant.

These tests skip rather than fail when no multimodal artifact is on this box,
because a box without the artifacts cannot answer the question and a fake
bundle would only test the fake.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO / "src"
MODELS = pathlib.Path("<models>")

# The exact line the fix introduced; the mutant restores the pre-fix ordering.
FIXED_ORDER = '_order = ("mlx_vlm", "mlx_lm") if _multimodal else ("mlx_lm", "mlx_vlm")'
PREFIX_ORDER = '_order = ("mlx_lm", "mlx_vlm")'


def _run(argv: list[str], cwd: pathlib.Path | None = None) -> subprocess.CompletedProcess:
    env = {"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}
    import os
    env = {**os.environ, **env}
    return subprocess.run([sys.executable, *argv], cwd=cwd or REPO,
                          capture_output=True, text=True, env=env, timeout=600)


def _a_multimodal_artifact(require_current_shim: bool = False) -> pathlib.Path | None:
    """A local artifact that declares a vision or audio tower.

    ``require_current_shim`` matters for the mutation test: not every artifact
    on this box has been re-bundled yet, and mutating a bundle that predates
    the fix proves nothing. Picking the FIRST multimodal directory silently
    skipped that test (it landed on a not-yet-rebundled GLM rung) — a test
    that skips is a test that is not running, which is the same class of
    problem as a gate nobody runs.
    """
    if not MODELS.is_dir():
        return None
    for path in sorted(MODELS.iterdir()):
        cfg = path / "config.json"
        model_py = path / "model.py"
        if not cfg.is_file() or not model_py.is_file():
            continue
        try:
            data = json.loads(cfg.read_text())
        except Exception:
            continue
        if not (data.get("vision_config") or data.get("audio_config")):
            continue
        if require_current_shim:
            try:
                if FIXED_ORDER not in model_py.read_text(encoding="utf-8"):
                    continue
            except Exception:
                continue
        return path
    return None


def test_vision_surface_is_wired_into_the_composite_gate():
    """`vqlab check` must RUN the vision arm, not merely mention it.

    Asserted against the source rather than a live run so it holds on a box
    with no artifacts: the point is that the gate list contains it.
    """
    src = (SRC / "vqlab" / "check_all.py").read_text(encoding="utf-8")
    assert "vision_smoke.py" in src, (
        "vqlab check must invoke vision_smoke.py — F153 shipped for three "
        "weeks because the vision question was never asked by the gate "
        "everyone runs"
    )
    gates_block = src[src.index("GATES = ["):src.index("def main")]
    assert "vision" in gates_block, "the vision arm must be a GATES entry"
    assert "--static" in gates_block, (
        "the composite must run the SURFACE arm (--static): it needs no "
        "weights, so it can run everywhere rather than only on a big box"
    )


def test_vision_arm_passes_a_correctly_bundled_multimodal_artifact():
    art = _a_multimodal_artifact(require_current_shim=True)
    if art is None:
        pytest.skip("no re-bundled multimodal artifact on this box")
    proc = _run(["-m", "vqlab.cli", "vision-smoke", str(art), "--static"])
    assert proc.returncode == 0, (
        f"vision-smoke failed on {art.name}, which should pass:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )
    assert "mlx_vlm" in proc.stdout, (
        "a multimodal artifact must resolve the MULTIMODAL base arch; "
        f"got:\n{proc.stdout}"
    )


def test_vision_arm_fails_the_historical_bug():
    """Mutation test: restore the pre-fix arch ordering, demand a FAIL.

    Without this, nothing distinguishes a working gate from one that prints
    PASS unconditionally — and an unconditional PASS is exactly what the old
    gate set was.
    """
    art = _a_multimodal_artifact(require_current_shim=True)
    if art is None:
        pytest.skip("no re-bundled multimodal artifact on this box to mutate")
    model_py = (art / "model.py").read_text(encoding="utf-8")

    with tempfile.TemporaryDirectory() as tmp:
        mutant = pathlib.Path(tmp) / "mutant"
        mutant.mkdir()
        for name in ("config.json", "model.safetensors.index.json"):
            if (art / name).is_file():
                shutil.copy2(art / name, mutant / name)
        (mutant / "model.py").write_text(
            model_py.replace(FIXED_ORDER, PREFIX_ORDER, 1), encoding="utf-8")

        proc = _run(["-m", "vqlab.cli", "vision-smoke", str(mutant), "--static"])

    assert proc.returncode != 0, (
        "the vision gate PASSED a bundle whose arch resolution is the exact "
        "pre-fix rule. That mutant is F153; a gate that cannot see it is the "
        f"gate we already had.\n{proc.stdout}\n{proc.stderr}"
    )
    combined = proc.stdout + proc.stderr
    assert "mlx_lm" in combined, (
        "the failure must NAME the text-only arch it resolved, or the next "
        f"person cannot act on it:\n{combined}"
    )


def test_text_only_artifacts_are_skipped_not_failed():
    """A text-only artifact must not be punished by a vision gate.

    Otherwise the composite becomes unusable for most of the fleet and gets
    routed around, which is how a gate quietly stops running.
    """
    if not MODELS.is_dir():
        pytest.skip("no model library on this box")
    text_only = None
    for path in sorted(MODELS.iterdir()):
        cfg = path / "config.json"
        if not cfg.is_file() or not (path / "model.py").is_file():
            continue
        try:
            data = json.loads(cfg.read_text())
        except Exception:
            continue
        if not (data.get("vision_config") or data.get("audio_config")):
            text_only = path
            break
    if text_only is None:
        pytest.skip("no text-only bundled artifact on this box")

    proc = _run(["-m", "vqlab.cli", "vision-smoke", str(text_only), "--static"])
    assert proc.returncode == 0, (
        f"vision-smoke must SKIP text-only {text_only.name}, not fail it:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )
