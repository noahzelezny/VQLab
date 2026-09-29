"""STALE vs FAIL (check-bundle): the current runtime must give BYTE-identical
output to every revision listed in runtime/equivalent_revisions.json, for
every fleet geometry, at decode and prefill N, fp16 and bf16 input. A bundle
carrying such a revision is merely STALE (slower, same numbers). Compared as
uint16 bit patterns; each revision is executed from its own git text with its
own baked defaults.
"""
import json
import pathlib
import subprocess
import types

import numpy as np
import mlx.core as mx
import pytest

from vqlab import vq_switch as VS
from vqlab import vq_pack as VP

ROOT = pathlib.Path(__file__).resolve().parents[1]
REG = json.loads((ROOT / "src/vqlab/runtime/equivalent_revisions.json").read_text())

# (d, K, packed, IN, OUT): every decode/prefill geometry class in the fleet map
GEOMS = [(4, 2048, True, 512, 96), (4, 512, True, 512, 96), (4, 256, False, 512, 96),
         (4, 256, True, 640, 80), (4, 8192, True, 512, 96), (4, 16384, True, 512, 64),
         (8, 16384, True, 2048, 64), (8, 16384, True, 1024, 64),
         (2, 1024, True, 512, 96), (2, 512, True, 512, 96), (2, 2048, True, 704, 64),
         (2, 256, False, 512, 96)]


def _old(rev):
    try:
        src = subprocess.run(["git", "show", f"{rev['commit']}:{rev['path']}"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout
    except Exception:
        pytest.skip(f"git text for {rev['commit']} unavailable")
    m = types.ModuleType("vq_switch_" + rev["commit"])
    exec(compile(src, m.__name__, "exec"), m.__dict__)
    return m


def _bits(a):
    return np.array(a.astype(mx.float16)).view(np.uint16)


@pytest.mark.parametrize("rev", REG["revisions"], ids=lambda r: r["commit"])
def test_current_runtime_byte_equal_to_certified_revision(rev):
    old = _old(rev)
    r = np.random.default_rng(11)
    E = 8
    for d, K, packed, IN, OUT in GEOMS:
        codes = r.integers(0, K, (E, OUT, IN // d)).astype(np.uint16 if K > 256 else np.uint8)
        cb = (r.standard_normal((K, d)) * 0.05).astype(np.float16)
        sc = (r.standard_normal((E, OUT, IN // 64)) * 0.1 + 1).astype(np.float16)
        kw = {}
        if packed:
            bits = VP.bits_for_k(K)
            codes = VP.pack(codes.astype(np.uint32), bits)
            kw = dict(pack_bits=bits, in_features=IN)
        for T, top in ((1, 8), (3, 10), (700, 8)):
            for dt in (mx.float16, mx.bfloat16):
                x = mx.array(r.standard_normal((T, 1, 1, IN)).astype(np.float32)).astype(dt)
                p = np.ones(E); p[0] = 6; p /= p.sum()
                idx = mx.array(r.choice(E, size=(T, top), p=p).astype(np.uint32))
                ys = [M.VQSwitchLinear(mx.array(codes), mx.array(cb), mx.array(sc),
                                       group_size=64, **kw)(x, idx) for M in (VS, old)]
                mx.eval(ys)
                assert np.array_equal(_bits(ys[0]), _bits(ys[1])), \
                    (rev["commit"], d, K, packed, IN, T, dt)
