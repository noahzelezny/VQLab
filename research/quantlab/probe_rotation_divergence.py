#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""Find the first layer where the rotated model stops being H-equivalent.

Runs baseline and rotated block-by-block (score_streaming's convention),
checking rel_err( h_rot , h_base @ H ) after embed and after every layer.
The first layer where error jumps names the mis-transformed component.
"""
import gc
import sys

import numpy as np
import mlx.core as mx

mx.set_cache_limit(8 << 30)  # compute on GPU like the scorer; params load on cpu stream

BASE = "/Volumes/Thunderbay SSD/Exo Models/Qwen--Qwen3.5-35B-A3B"
ROT = "/Volumes/Thunderbay SSD/Exo Models/rotlab-35B-rotated-bf16"
NTOK = 64


def hadamard(n):
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return mx.array((h / np.sqrt(n)).astype(np.float32))


from mlx_lm.utils import load

def get_core(m):
    core = m
    for name in ("language_model", "model"):
        while hasattr(core, name):
            core = getattr(core, name)
    return core

with mx.stream(mx.cpu):
    mb, tok, _ = load(BASE, lazy=True, return_config=True)
    mr, _, _ = load(ROT, lazy=True, return_config=True)
cb, cr = get_core(mb), get_core(mr)
H = hadamard(1024 * 2)

text = open("referee/referee_corpus.txt").read()[:4000]
toks = mx.array(tok.encode(text))[:NTOK][None]


def relerr(a, b):
    a = a.astype(mx.float32)
    b = b.astype(mx.float32)
    return float((mx.linalg.norm(a - b) / (mx.linalg.norm(b) + 1e-9)).item())


with mx.stream(mx.cpu):
    mx.eval(cb.embed_tokens.parameters(), cr.embed_tokens.parameters())
hb = cb.embed_tokens(toks)
hr = cr.embed_tokens(toks)
mx.eval(hb, hr)
print(f"embed rel_err {relerr(hr, hb @ H):.5f}", flush=True)

for i in range(len(cb.layers)):
    bb, br = cb.layers[i], cr.layers[i]
    mask = None if bb.is_linear else "causal"
    with mx.stream(mx.cpu):
        mx.eval(bb.parameters(), br.parameters())
    bb.eval(); br.eval()
    hb = bb(hb, mask=mask, cache=None)
    hr = br(hr, mask=mask, cache=None)
    mx.eval(hb, hr)
    kind = "lin " if bb.is_linear else "full"
    print(f"layer {i:2d} {kind} rel_err {relerr(hr, hb @ H):.5f}", flush=True)
    cb.layers[i] = cr.layers[i] = None
    del bb, br
    gc.collect()
    mx.clear_cache()
