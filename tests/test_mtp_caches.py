"""Rollback correctness, and the gate that decides whether a family may use
free (non-copying) snapshots.

Per CONTRIBUTING: a gate is only trusted once it FAILS on a known-bad input
and PASSES on a known-good one, so both caches below are exercised.
"""
import mlx.core as mx
import pytest

from vqlab.mtp.caches import check_snapshot_semantics, restore, snapshot


class AttnCache:
    """Trimmable attention cache, as mlx-lm's KVCache is."""

    def __init__(self):
        self.keys = []
        self.offset = 0

    def update(self, tok):
        self.keys = self.keys + [tok]
        self.offset += 1

    def trim(self, n):
        assert n <= len(self.keys)
        self.keys = self.keys[: len(self.keys) - n]
        self.offset -= n
        return n


class ReassigningCache:
    """qwen4_exp's shape: the slot is REPLACED, never written into."""

    def __init__(self):
        self.cache = [mx.zeros((4,))]
        self.offset = 0

    def update(self, tok):
        self.cache = [self.cache[0] * 0.5 + float(tok)]
        self.offset += 1


class MutatingCache(ReassigningCache):
    """The hazard: same interface, writes state in place."""

    def update(self, tok):
        self.cache[0][self.offset % 4] = float(tok)
        self.offset += 1


def test_attention_rolls_back_by_the_offset_delta():
    """The bug this exists to prevent: trimming a hardcoded 1 after a 2-token
    forward leaves a stale key while the state caches roll back 2."""
    a = AttnCache()
    for t in (1, 2, 3):
        a.update(t)
    snap = snapshot([a])
    a.update(4)
    a.update(5)
    restore([a], snap)
    assert a.offset == 3
    assert a.keys == [1, 2, 3]     # not [1, 2, 3, 4]: the delta was 2


def test_attention_rollback_is_a_noop_when_nothing_advanced():
    a = AttnCache()
    a.update(1)
    snap = snapshot([a])
    restore([a], snap)
    assert a.offset == 1 and a.keys == [1]


def test_restore_refuses_a_cache_that_went_backwards():
    a = AttnCache()
    for t in (1, 2, 3):
        a.update(t)
    snap = snapshot([a])
    a.trim(2)
    with pytest.raises(RuntimeError, match="BACKWARDS"):
        restore([a], snap)


def test_state_cache_restores_exactly():
    c = ReassigningCache()
    for t in (1, 2):
        c.update(t)
    snap = snapshot([c])
    before = mx.array(c.cache[0])
    c.update(9)
    c.update(9)
    restore([c], snap)
    assert c.offset == 2
    assert bool(mx.all(c.cache[0] == before).item())


def test_snapshot_refuses_an_unrecognised_cache():
    class Opaque:
        pass

    with pytest.raises(TypeError, match="rollback cannot be proven correct"):
        snapshot([Opaque()])


def test_semantics_gate_passes_on_a_reassigning_cache():
    """Known-good: qwen4_exp's shape earns cache_semantics='reassign'."""
    c = ReassigningCache()
    c.update(1)
    assert check_snapshot_semantics([c], lambda: c.update(2)) is True


def test_semantics_gate_fails_on_an_in_place_cache():
    """Known-bad: mlx arrays DO support item assignment, so a family that
    writes state in place would corrupt the very reference a free snapshot
    holds. The gate must catch it; such a family must copy."""
    c = MutatingCache()
    c.update(1)
    assert check_snapshot_semantics([c], lambda: c.update(2)) is False


def test_copying_snapshot_survives_an_in_place_cache():
    """And copy=True is the cure, so a new family is safe before it is
    measured."""
    c = MutatingCache()
    c.update(1)
    snap = snapshot([c], copy=True)
    before = mx.array(snap[0][2][0])
    c.update(7)
    restore([c], snap)
    assert bool(mx.all(c.cache[0] == before).item())
