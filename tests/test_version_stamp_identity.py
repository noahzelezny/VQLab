"""Untangle step 9a moved four modules (core/families -> family/registry,
core/numerics -> family/version_stamp, core/runtime_load -> family/model_loader,
family/arch -> core/arch_finder). The
old names are the SAME module objects, and the numerics stamp a score writes
is byte-identical whichever name built it: no stamped field names a module."""
from __future__ import annotations

import importlib
import json
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vqlab import _layout  # noqa: E402,F401

MOVED = {
    "vqlab.core.families": "vqlab.family.registry",
    "families": "vqlab.family.registry",
    "vqlab.families": "vqlab.family.registry",
    "vqlab.core.numerics": "vqlab.family.version_stamp",
    "numerics": "vqlab.family.version_stamp",
    "vqlab.numerics": "vqlab.family.version_stamp",
    "vqlab.core.version_stamp": "vqlab.family.version_stamp",
    "version_stamp": "vqlab.family.version_stamp",
    "vqlab.core.runtime_load": "vqlab.family.model_loader",
    "vqlab.runtime_load": "vqlab.family.model_loader",
    "runtime_load": "vqlab.family.model_loader",
    "vqlab.family.arch": "vqlab.core.arch_finder",
}


@pytest.mark.parametrize("old,new", sorted(MOVED.items()))
def test_old_name_is_same_module(old, new):
    assert importlib.import_module(old) is importlib.import_module(new)


def test_old_attribute_names_still_answer():
    import vqlab.core.arch_finder as A
    import vqlab.core.numerics as N
    import vqlab.family.registry as R
    from vqlab.family import model_loader as runtime_load
    assert A.install is A.install_arch_finder
    assert runtime_load.FAMILY is R.FAMILY
    import vqlab.core.stamp_fields as F
    assert (N.describe, N.diff, N.COMPARED) == (F.describe, F.diff, F.COMPARED)
    assert N.scorer_variant("deepseek_v4") == R.scorer_variant("deepseek_v4")


CASES = [("deepseek_v4", "deepseek_v4"), ("qwen4_exp", "qwen4_exp"),
         ("glm5_next", "glm5_next"), ("qwen3_5_moe", None), (None, None)]


def test_stamp_identical_under_old_and_new_name():
    old = importlib.import_module("vqlab.core.numerics")
    new = importlib.import_module("vqlab.family.version_stamp")
    for mt, fam in CASES:
        a, b = old.build(mt, family=fam), new.build(mt, family=fam)
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
        blob = json.dumps(a)
        assert "numerics" not in blob and "version_stamp" not in blob


def test_stamp_cli_byte_identical_under_old_and_new_name():
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    out = [subprocess.run([sys.executable, "-m", m, "--model-type", "deepseek_v4"],
                          capture_output=True, text=True, env=env, check=True).stdout
           for m in ("vqlab.core.numerics", "vqlab.family.version_stamp")]
    assert out[0] == out[1] and json.loads(out[0])["model_type"] == "deepseek_v4"
