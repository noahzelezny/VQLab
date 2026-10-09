"""STALE vs FAIL (check-bundle): the current runtime must give BYTE-identical
output to every revision listed in runtime/equivalent_revisions.json, for
every fleet geometry, at decode and prefill N, fp16 and bf16 input. A bundle
carrying such a revision is merely STALE (slower, same numbers). Compared as
uint16 bit patterns; each revision is executed from its own git text with its
own baked defaults.
"""
import json
import pathlib
import types

import numpy as np
import mlx.core as mx
import pytest

from vqlab import vq_switch as VS
from vqlab import vq_pack as VP


@pytest.fixture(autouse=True)
def _legacy_prefill_flags(monkeypatch):
    """These tests certify byte-equality against reference paths / earlier
    revisions that predate the rebundle defaults (VQ_FUSED_MAX_N 512,
    VQ_GEMMSEG_ACCS=1, VQ_GEMMSEG_CVEC=1; F200-F204). Those defaults change
    numerics for N in 512..4096 by design, so pin the legacy values here."""
    monkeypatch.setattr(VS, "VQ_FUSED_MAX_N", 4096)
    monkeypatch.setattr(VS, "_GEMMSEG_ACCS", False)
    monkeypatch.setattr(VS, "_GEMMSEG_CVEC", False)

ROOT = pathlib.Path(__file__).resolve().parents[1]
REG = json.loads((ROOT / "src/vqlab/runtime/equivalent_revisions.json").read_text())

# (d, K, packed, IN, OUT): every decode/prefill geometry class in the fleet map
GEOMS = [(4, 2048, True, 512, 96), (4, 512, True, 512, 96), (4, 256, False, 512, 96),
         (4, 256, True, 640, 80), (4, 8192, True, 512, 96), (4, 16384, True, 512, 64),
         (8, 16384, True, 2048, 64), (8, 16384, True, 1024, 64),
         (2, 1024, True, 512, 96), (2, 512, True, 512, 96), (2, 2048, True, 704, 64),
         (2, 256, False, 512, 96)]


def _text(rev):
    return (ROOT / "src/vqlab/runtime" / rev["file"]).read_text()


def _old(rev):
    src = _text(rev)
    m = types.ModuleType("vq_switch_" + rev["sha256"][:12])
    exec(compile(src, m.__name__, "exec"), m.__dict__)
    return m


def _bits(a):
    return np.array(a.astype(mx.float16)).view(np.uint16)


SWITCH = [r for r in REG["revisions"] if r["path"].endswith("vq_switch.py")]
OTHER = [r for r in REG["revisions"] if not r["path"].endswith("vq_switch.py")]


def _code_tokens(text):
    """Python tokens with comments and blank-line tokens dropped: two texts
    with equal token streams run identically."""
    import io
    import tokenize
    skip = {tokenize.COMMENT, tokenize.NL}
    toks = [(t.type, t.string) for t in tokenize.generate_tokens(io.StringIO(text).readline)
            if t.type not in skip]
    # Narrow normalization: `os.environ.get("VQ_X", 96)` and
    # `os.environ.get("VQ_X", "96")` are the same behaviour once wrapped in
    # int()/float(), and runtime_profile only tracks the string spelling. A
    # numeric literal that is the 2nd argument of os.environ.get("VQ_...", .)
    # is compared as its quoted form.
    out = []
    for i, (ty, s) in enumerate(toks):
        if (ty == tokenize.NUMBER and i >= 7 and toks[i - 1][1] == ","
                and toks[i - 2][1].startswith('"VQ_') and toks[i - 3][1] == "("
                and [x[1] for x in toks[i - 6:i - 3]] == ["environ", ".", "get"]):
            ty, s = tokenize.STRING, '"%s"' % s
        out.append((ty, s))
    return out


@pytest.mark.parametrize("rev", OTHER, ids=lambda r: r["sha256"][:12] + ":" + r["path"].rsplit("/", 1)[-1])
def test_other_runtime_files_differ_only_in_comments(rev):
    """Non-kernel runtime files are certified only when the change is
    comment-only: the token stream (docstrings included) must be identical."""
    old = _text(rev)
    new = (ROOT / rev["path"]).read_text()
    assert _code_tokens(old) == _code_tokens(new)


@pytest.mark.parametrize("rev", SWITCH, ids=lambda r: r["sha256"][:12])
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
                    (rev["sha256"][:12], d, K, packed, IN, T, dt)
