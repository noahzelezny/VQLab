"""family/arch serves Knurlogic's architectures as mlx_lm.models.<name>,
including the ones stock mlx-lm lacks; VQLAB_VENDORED_ARCH=0 turns it off."""
import os
import subprocess
import sys

import pytest

pytest.importorskip("mlx_lm")

PROBE = ("import importlib, vqlab; from vqlab.family import arch;"
         "m = importlib.util.find_spec('mlx_lm.models.deepseek_v4');"
         "print(m.origin if m else None)")


def _origin(env_value):
    env = dict(os.environ, VQLAB_VENDORED_ARCH=env_value)
    return subprocess.run([sys.executable, "-c", PROBE], env=env, capture_output=True,
                          text=True, check=True).stdout.strip()


def test_vendored_serves_deepseek_v4():
    assert _origin("1").endswith("vqlab/family/arch/deepseek/deepseek_v4.py")


def test_off_switch_falls_back_to_stock():
    o = _origin("0")
    assert "vqlab/family/arch" not in o


def test_every_source_exists():
    from vqlab.family import arch
    assert all(p.is_file() for p in arch.SOURCES.values())
