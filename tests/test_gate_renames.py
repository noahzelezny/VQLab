"""gate/kernel_reference.py: the exact reference dequant-matmul."""

import pytest



def test_reference_dequant_matmul():
    mx = pytest.importorskip("mlx.core")
    from vqlab.gate.kernel_reference import reference
    codes = mx.array([[[0, 1]], [[1, 1]]], dtype=mx.uint8)       # E2 OUT1 NSUB2
    cb = mx.array([[1.0, 0.0], [0.0, 2.0]])                       # K2 D2
    sc = mx.array([[[2.0]], [[1.0]]])                             # G4
    x = mx.array([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    y = reference(x, mx.array([0, 1]), codes, cb, sc, 2, 4)
    assert y.tolist() == [[6.0], [4.0]]
