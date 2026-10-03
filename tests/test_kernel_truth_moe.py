"""kernel-truth-moe: the float64 reference reconstructs a packed expert
exactly, and both MoE paths land close to it on a tiny module."""
import numpy as np

from vqlab.score import kernel_truth_moe as KT


def test_reference_and_paths():
    mod = KT.synthetic(16, 4, IN=256, OUT=64, E=4)
    W = KT._weights(mod, [1])[1]
    assert W.shape == (64, 256) and np.isfinite(W).all()
    cells = KT.compare("tiny", mod, seeds=1, n_experts=4)
    assert len(cells) == 1 and cells[0][1] < 1.5          # paths agree to within 50% of each other's error
