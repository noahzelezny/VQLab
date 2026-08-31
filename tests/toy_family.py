"""A tiny model + drafting head that satisfies the same seam as a real one.

The point is to test the loop, not an architecture: this model's logits are a
deterministic function of the ENTIRE committed prefix, held in both a
trimmable attention-style cache and a reassigning state cache. So a rollback
that trims the wrong number of positions, or that fails to restore the state
cache, changes the output — which is exactly what the loop tests assert
against.

Registering this family from a test file is also the check that the registry
is genuinely a table: nothing outside registry.py names an architecture.
"""
from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from vqlab.mtp.registry import FamilySpec, register, unregister

VOCAB = 6
DIM = 4
FAMILY = "toy_mtp"

mx.random.seed(0)
_W = mx.random.normal((DIM, VOCAB)) * 1.5


class ToyAttnCache:
    def __init__(self):
        self.keys = []
        self.offset = 0

    def update(self, tok):
        self.keys = self.keys + [tok]
        self.offset += 1

    def trim(self, n):
        self.keys = self.keys[: len(self.keys) - n]
        self.offset -= n
        return n


class ToyDraftCache(ToyAttnCache):
    """The head's own cache. Named on the arch module, resolved through the
    registry's `draft_cache` field."""


class ToyStateCache:
    """Reassigns its slot, as qwen4_exp does."""

    def __init__(self):
        self.cache = [mx.zeros((DIM,))]
        self.offset = 0

    def update(self, tok):
        self.cache = [self.cache[0] * 0.4 + float(tok + 1)]
        self.offset += 1


def _hidden(keys, state):
    acc = 0
    for t in keys:
        acc = (acc * 131 + t + 1) % 9973
    base = mx.array([math.sin(acc * (k + 1) * 0.7391) for k in range(DIM)])
    return base + 0.05 * state


class ToyMixer(nn.Module):
    """Stands in for the module whose input is the pre-lm_head activation."""

    def __call__(self, x):
        return x


class ToyCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.hyper_connection_mixer = ToyMixer()


class ToyModel(nn.Module):
    model_type = FAMILY

    def __init__(self):
        super().__init__()
        self.model = ToyCore()

    def make_cache(self):
        return [ToyAttnCache(), ToyStateCache()]

    def __call__(self, tokens, cache=None):
        rows = []
        for i in range(tokens.shape[1]):
            tok = int(tokens[0, i].item())
            cache[0].update(tok)
            cache[1].update(tok)
            rows.append(_hidden(cache[0].keys, cache[1].cache[0]))
        h = mx.stack(rows)[None]
        h = self.model.hyper_connection_mixer(h)
        return h @ _W


class ToyHead:
    """Drafts token t+2 from (hidden at t, token t+1).

    modes:
      stubborn    always drafts the same token; blind, mostly wrong
      biased      a real function of the inputs; sometimes right
      oracle      returns a known-correct token, so every step accepts
      antioracle  returns a known-WRONG token, so every step rejects
    """

    def __init__(self, mode="biased", oracle=None):
        self.mode = mode
        self.oracle = oracle or []
        self.calls = 0

    def draft_logits(self, h_row, nxt_id, cache=None):
        if cache is not None:
            cache.update(int(nxt_id[0, 0].item()))
        i = self.calls
        self.calls += 1
        if self.mode in ("oracle", "antioracle"):
            pick = self.oracle[2 * i + 1]
            if self.mode == "antioracle":
                pick = (pick + 1) % VOCAB
            return mx.where(mx.arange(VOCAB) == pick, 30.0, 0.0)[None, None]
        if self.mode == "stubborn":
            return mx.where(mx.arange(VOCAB) == VOCAB - 1, 30.0,
                            0.0)[None, None]
        return (h_row @ _W) * 2.0 + mx.arange(VOCAB) * 0.1


SPEC = FamilySpec(
    name=FAMILY,
    head="toy_family:ToyHead",
    capture="hyper_connection_mixer",
    draft_cache="ToyDraftCache",
    sidecar_name="toy-head.safetensors",
    cache_semantics="reassign",
)


def install():
    return register(SPEC, replace=True)


def remove():
    unregister(FAMILY)
