"""d2 fused dispatch must not walk into the E134 kernel-load cliff.

All three fused d2 kernels cache `half2 cb[MAX_K]` + `half2 xs[MAX_NSUB]` =
(K + NSUB) * 4 B in threadgroup memory. Past Apple's 32,768 B cap the kernel
fails to LOAD, and a load failure is not something a caller can turn into an
answer -- which is exactly why the d4 side grew _d4_tg_fits and its
device-codebook fallback. The d2 side had no such check: found while auditing
the gemma divergence (docs/GEMMA-DIVERGENCE-2026-09-07.md, "Latent").

No shipped artifact reaches it (max d2 K is 2048 -> 13,824 B at the widest
layer), so this suite drives SYNTHETIC K=4096 shapes: it is a ratchet against
the cliff being reintroduced, not a regression test for a live bug.

There is no device-cb d2 kernel to fall back to, so the graceful path is the
decode path (vq_decode caches nothing in threadgroup memory and is
capacity-independent at any K). The contract asserted here is therefore
twofold: the oversized shape must not crash, AND the shape just UNDER the cap
must still take the fast fused kernel and agree with the fallback -- a guard
that silently swallowed every d2 call into the slow path would pass a
crash-only test.
"""
import pathlib
import sys

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src" / "vqlab"))
import vq_pack  # noqa: E402
import vq_switch as V  # noqa: E402


def _d2_case(E, OUT, IN, K, G, N, packed=0, seed=0):
    rng = np.random.default_rng(seed)
    NSUB = IN // 2
    codes = rng.integers(0, K, size=(E, OUT, NSUB),
                         dtype=np.uint16 if K > 256 else np.uint8)
    c = mx.array(vq_pack.pack(codes, packed)) if packed else mx.array(codes)
    return (mx.array(rng.normal(0, 1, size=(N, IN)).astype(np.float16)),
            mx.array(rng.integers(0, E, size=(N,)).astype(np.uint32)),
            c,
            mx.array(rng.normal(0, 1, size=(K, 2)).astype(np.float16)),
            mx.array(rng.normal(0, 0.05, size=(E, OUT, IN // G))
                     .astype(np.float16)))


def test_d2_tg_fits_matches_the_kernels_allocation():
    # (K + NSUB) * 4 B, the literal allocation in _SRC_FUSED_D2 /
    # _SRC_FUSED_D2_U32 / _SRC_FUSED_PACKED_D2.
    cap = V._TG_CAP_BYTES
    assert V._d2_tg_fits(cap // 4 - 1, 1)
    assert V._d2_tg_fits(cap // 4, 0)
    assert not V._d2_tg_fits(cap // 4, 1)
    # the widest shipped d2 shape (K2048, IN 2816 -> NSUB 1408) clears it
    assert V._d2_tg_fits(2048, 1408)
    # K=4096 lands EXACTLY on the cap at NSUB=4096 (8192 * 4 = 32,768) -- it
    # is the next subvector that tips it, which is why the oversized shape
    # below uses a wider IN rather than "K=4096 is obviously too big".
    assert V._d2_tg_fits(4096, 4096)
    assert not V._d2_tg_fits(4096, 4097)


# (E, OUT, IN, K, G) with d=2. K=4096, NSUB=8192 -> (4096+8192)*4 = 49,152 B,
# 1.5x the cap, so the fused kernel cannot load.
OVERSIZED = (2, 128, 16384, 4096, 64)


@pytest.mark.parametrize("packed", [0, 12])   # K=4096 needs 12-bit codes
def test_oversized_d2_does_not_crash_the_kernel_loader(packed):
    E, OUT, IN, K, G = OVERSIZED
    assert not V._d2_tg_fits(K, IN // 2), "shape no longer exercises the cliff"
    args = _d2_case(E, OUT, IN, K, G, N=4, packed=packed)
    y = V._fused(*args, pack_bits=packed)
    mx.eval(y)
    assert y.shape == (4, OUT)
    assert bool(mx.all(mx.isfinite(y)))


@pytest.mark.parametrize("packed", [0, 12])
def test_oversized_d2_fallback_agrees_with_decode(packed):
    # The fallback IS the decode path, so this pins that _fused routes there
    # rather than to some other kernel that merely happens to load.
    E, OUT, IN, K, G = OVERSIZED
    args = _d2_case(E, OUT, IN, K, G, N=4, packed=packed)
    got = np.array(V._fused(*args, pack_bits=packed))
    want = np.array(V._fused_decode_fallback(*args, packed))
    assert np.array_equal(got, want)


@pytest.mark.parametrize("packed", [0, 11])
def test_under_cap_d2_still_takes_the_fused_kernel(packed):
    # Guard against an over-broad fits check quietly demoting every d2 call:
    # a shipped-shape d2 must still hit the threadgroup kernel, and its answer
    # must match the decode path to fp16 rounding.
    E, OUT, IN, K, G = 2, 128, 2816, 2048, 64
    assert V._d2_tg_fits(K, IN // 2)
    args = _d2_case(E, OUT, IN, K, G, N=4, packed=packed)
    fused = np.array(V._fused(*args, pack_bits=packed)).astype(np.float32)
    decode = np.array(V._fused_decode_fallback(*args, packed)).astype(np.float32)
    denom = np.maximum(np.abs(decode).max(), 1e-6)
    assert np.abs(fused - decode).max() / denom < 5e-3
