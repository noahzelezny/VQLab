"""Clamp counting for act-stats: the counter and the house-corpus ids.

In core because a family plugin (deepseek_v4's act_stats hook) uses them and
the score stage's act-stats CLI drives that hook; family may not import score.
"""
from __future__ import annotations


class ClampCounter:
    """Counts SwiGLU clamp events the way DeepSeek's reference clamps:
    gate is clamped from ABOVE only, up on BOTH sides, at one limit."""

    def __init__(self, limit: float):
        self.limit = float(limit)
        self.calls = self.elems = self.gate_over = self.up_over = self.any_over = 0

    def add(self, gate, up) -> None:
        import mlx.core as mx
        go, uo = gate > self.limit, mx.abs(up) > self.limit
        n = [mx.sum(go), mx.sum(uo), mx.sum(go | uo)]
        mx.eval(n)
        self.calls += 1
        self.elems += gate.size
        self.gate_over += int(n[0].item())
        self.up_over += int(n[1].item())
        self.any_over += int(n[2].item())

    def result(self) -> dict:
        return {"limit": self.limit, "calls": self.calls, "elements": self.elems,
                "gate_over": self.gate_over, "up_abs_over": self.up_over,
                "any_over": self.any_over,
                "frac_any_over": self.any_over / max(1, self.elems)}


def corpus_ids(tok, name: str, tokens: int) -> list:
    """N+1 ids of a house corpus (the scorer drops the last), BOS-led."""
    from vqlab import _layout
    ids = tok.encode(_layout.corpus(name).read_text())[: tokens + 1]
    bos = getattr(tok, "bos_token_id", None)
    if bos is not None and (not ids or ids[0] != bos):
        ids = [bos] + ids[:tokens]
    return ids
