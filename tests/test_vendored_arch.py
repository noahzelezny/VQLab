"""vqlab.family.arch loads Knurlogic's architectures as mlx_lm.models.<name>
(one copy: VQLab depends on knurlogic), including the ones stock mlx-lm
lacks; VQLAB_VENDORED_ARCH=0 turns it off."""
import os
import subprocess
import sys

import pytest

pytest.importorskip("mlx_lm")
pytest.importorskip("knurlogic")

PROBE = ("import importlib, vqlab;"
         "m = importlib.import_module('mlx_lm.models.deepseek_v4');"
         "print(m.__file__)")


def _origin(env_value):
    env = dict(os.environ, VQLAB_VENDORED_ARCH=env_value)
    r = subprocess.run([sys.executable, "-c", PROBE], env=env, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def test_knurlogic_serves_deepseek_v4():
    assert "knurlogic/engine/families/deepseek/architecture/deepseek_v4.py" in _origin("1")


def test_off_switch_falls_back_to_stock():
    o = _origin("0")
    assert o is None or "knurlogic" not in o      # stock mlx-lm 0.32 has none


def test_no_cloned_copy_in_vqlab():
    import pathlib
    import vqlab
    assert not (pathlib.Path(vqlab.__file__).parent / "family" / "arch").is_dir()
