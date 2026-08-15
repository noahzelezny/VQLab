#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""E26 diagnosis v2: time the REAL training backward, piecewise, solo M4.

probe v1: one tail layer fwd+bwd = 0.6s (innocent). This chains the actual
loss path — tail-6 layers -> norm -> lm_head -> take_along_axis top-1024 ->
KL — and times: (a) layers-only backward, (b) +norm+head backward,
(c) the full loss backward. Whichever jump crosses ~2s names the watchdog
victim. Unsharded (12G tail + 2G head fits the M4 easily).
"""
import time

import mlx.core as mx
from mlx.utils import tree_flatten


MODEL = "/Users/noahzelezny/.exo/models/TheDrainFlorist--Qwen3.5-397B-A17B-struct6-tail3x3"

from mlx_lm.utils import load
with mx.stream(mx.cpu):
    model, _, _ = load(MODEL, lazy=True, return_config=True)
core = model
for name in ("language_model", "model"):
    while hasattr(core, name):
        core = getattr(core, name)

tail = core.layers[-6:]
head_norm = core.norm
lm_head = getattr(model, "lm_head", None) or getattr(
    getattr(model, "language_model", model), "lm_head", None)
with mx.stream(mx.cpu):
    for l in tail:
        mx.eval(l.parameters())
    mx.eval(head_norm.parameters(), lm_head.parameters())
print(f"materialized {mx.get_active_memory() / 1024**3:.1f}G", flush=True)


def unfreeze(_, m):
    if (hasattr(m, "bits") and hasattr(m, "group_size")
            and getattr(m, "mode", None) == "affine" and m.bits < 8):
        m.unfreeze(keys=["scales", "biases"], recurse=False)


for l in tail:
    l.freeze()
    l.apply_to_modules(unfreeze)
    l.train()

S = 512
x = mx.random.normal((1, S, 4096)).astype(mx.bfloat16)
t_idx = mx.random.randint(0, 248000, (1, S, 1024))
t_lp = mx.random.normal((1, S, 1024)).astype(mx.bfloat16)

params = {i: l.trainable_parameters() for i, l in enumerate(tail)}
n = sum(v.size for _, v in tree_flatten(params))
print(f"trainable {n / 1e6:.0f}M", flush=True)


def run(label, fn):
    t0 = time.time()
    _, g = mx.value_and_grad(fn)(params)
    mx.eval(g)
    print(f"{label}: {time.time() - t0:.2f}s", flush=True)


def layers_only(p):
    h = x
    for i, l in enumerate(tail):
        l.update(p[i])
        h = l(h, mask=("causal" if not l.is_linear else None), cache=None)
    return h.astype(mx.float32).sum()


def plus_head(p):
    h = x
    for i, l in enumerate(tail):
        l.update(p[i])
        h = l(h, mask=("causal" if not l.is_linear else None), cache=None)
    logits = lm_head(head_norm(h))
    return logits.astype(mx.float32).sum()


def full_loss(p):
    h = x
    for i, l in enumerate(tail):
        l.update(p[i])
        h = l(h, mask=("causal" if not l.is_linear else None), cache=None)
    logits = lm_head(head_norm(h))
    logits = mx.take_along_axis(logits, t_idx, axis=-1).astype(mx.float32)
    t = t_lp.astype(mx.float32)
    ls = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    ts = t - mx.logsumexp(t, axis=-1, keepdims=True)
    return (mx.exp(ts) * (ts - ls)).sum(-1).mean()


run("tail-6 layers only     ", layers_only)
run("tail-6 + norm + lm_head", plus_head)
run("full KL loss           ", full_loss)
print("done", flush=True)
