"""The dense kernels' output must not depend on rows per threadgroup.

`_DENSE_ROWS_TG` is 16 because some Apple GPUs cap the register-heavy DEVX /
tiled pipelines below 1024 threads (832 on the GitHub macOS arm64 runner), so
a 32-row (1024-thread) threadgroup fails there. Each row is one simdgroup's
independent reduction, so the row count must be a pure launch parameter:
byte-identical output at 8, 16 and 32 rows, every dense kernel family.
"""
import itertools
import os

import numpy as np
import pytest

import mlx.core as mx

import vq_pack
import vq_switch as V

CASES = [(2, 512, 0), (2, 512, 9), (2, 4096, 0), (4, 256, 0), (4, 4096, 12)]


def _layer(OUT, IN, K, D, packed, N, seed=1234):
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, K, size=(OUT, IN // D), dtype=np.uint16 if K > 256 else np.uint8)
    c = mx.array(vq_pack.pack(codes[None], packed)[0]) if packed else mx.array(codes)
    return (mx.array(rng.normal(0, 1, (N, IN)).astype(np.float16)), c,
            mx.array(rng.normal(0, 1, (K, D)).astype(np.float16)),
            mx.array(rng.normal(0, 0.05, (OUT, IN // 64)).astype(np.float16)))


@pytest.mark.parametrize("D,K,packed", CASES)
@pytest.mark.parametrize("devx,tiled", list(itertools.product([0, 1], ["0", "1"])))
def test_dense_output_independent_of_rows_per_threadgroup(D, K, packed, devx, tiled,
                                                          monkeypatch):
    monkeypatch.setenv("VQ_DENSE_TILED", tiled)
    monkeypatch.setattr(V, "_DENSE_DEVX", devx)
    x, c, cb, sc = _layer(72, 2048, K, D, packed, N=3)
    out = {}
    for rows in (8, 16, 32):
        monkeypatch.setattr(V, "_DENSE_ROWS_TG", rows)
        try:
            y = V._dense_fused(x, c, cb, sc, pack_bits=packed, in_features=2048)
            mx.eval(y)
        except ValueError as e:
            # A GPU whose pipeline limit is below rows*32 threads cannot run
            # that launch at all; that is exactly why the default is 16.
            assert rows * 32 > 512 and "threadgroup" in str(e), e
            continue
        out[rows] = np.array(y.view(mx.uint16))
    assert 16 in out, "the default launch (16 rows) must run on every GPU"
    for rows, y in out.items():
        assert np.array_equal(y, out[16]), f"{rows} rows differs from 16"


def test_default_is_16_rows():
    if "VQ_DENSE_ROWS_TG" not in os.environ:
        assert V._DENSE_ROWS_TG == 16
