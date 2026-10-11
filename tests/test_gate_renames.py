"""gate/kernel_reference.py, and the experiment-numbered gate files renamed
to what they do (old names still resolve)."""
import importlib.util

import pytest

from vqlab import _layout


@pytest.mark.parametrize("old,new", [("e134_accept", "kernel_accept"),
                                     ("verify_e134_devcb", "verify_dev_codebook")])
def test_old_gate_names_resolve(old, new):
    names = _layout._names()
    assert importlib.util.find_spec(old) is None      # bare tool names: gone (R1)
    for spelling in (f"vqlab.{old}", f"vqlab.gate.{old}"):
        assert names[spelling] == f"vqlab.gate.{new}"
        assert importlib.util.find_spec(spelling) is not None
    assert _layout.find(f"{old}.py").name == f"{new}.py"


def test_reference_dequant_matmul():
    mx = pytest.importorskip("mlx.core")
    from vqlab.gate.kernel_reference import reference
    codes = mx.array([[[0, 1]], [[1, 1]]], dtype=mx.uint8)       # E2 OUT1 NSUB2
    cb = mx.array([[1.0, 0.0], [0.0, 2.0]])                       # K2 D2
    sc = mx.array([[[2.0]], [[1.0]]])                             # G4
    x = mx.array([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    y = reference(x, mx.array([0, 1]), codes, cb, sc, 2, 4)
    assert y.tolist() == [[6.0], [4.0]]
