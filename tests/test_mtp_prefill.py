"""Chunked prefill: the chunk width is a memory knob, never an output knob.

The failure mode this guards is arithmetic at the chunk edges. The trunk is
chunked over ids[:, :n-1] and the head is seeded per chunk with
(h_i..h_{end-1}, x_{i+1}..x_{end}); an off-by-one at a boundary would drop or
duplicate one head row, which does NOT raise -- it only shifts the head's
rotary positions and degrades acceptance. So these tests assert on the exact
(hidden, token) stream the head is fed, not just on the tokens emitted.

Prompt lengths deliberately straddle the width: shorter than one chunk, an
exact multiple, and longer with a ragged tail.
"""
import mlx.core as mx
import pytest

import toy_family as toy
from vqlab.mtp import mtp_stream_generate, prefill_chunk_size
from vqlab.mtp.loop import DEFAULT_PREFILL_CHUNK


class Tok:
    eos_token_ids: set = set()

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


@pytest.fixture(autouse=True)
def family():
    toy.install()
    yield
    toy.remove()


class RecordingHead(toy.ToyHead):
    """A ToyHead that records everything `advance` is fed during the seed."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.seed_hidden = []
        self.seed_ids = []

    def advance(self, h_row, nxt_id, cache):
        self.seed_hidden.append(mx.array(h_row))
        self.seed_ids.extend(int(t) for t in nxt_id[0].tolist())
        return super().advance(h_row, nxt_id, cache)

    def seed_stream(self):
        if not self.seed_hidden:
            return mx.zeros((1, 0, 1)), []
        return mx.concatenate(self.seed_hidden, axis=1), list(self.seed_ids)


def prompt_of(n):
    return mx.array([[(i % (toy.VOCAB - 1)) + 1 for i in range(n)]])


def run(prompt, step, n=8, head=None):
    head = head or RecordingHead("biased")
    toks = [r.token for r in mtp_stream_generate(
        toy.ToyModel(), Tok(), prompt, head, max_tokens=n,
        prefill_step_size=step)]
    return toks, head


# --- the resolver -----------------------------------------------------------

def test_default_chunk_is_2048():
    assert prefill_chunk_size() == DEFAULT_PREFILL_CHUNK == 2048


def test_env_overrides_the_default(monkeypatch):
    monkeypatch.setenv("VQLAB_PREFILL_CHUNK", "512")
    assert prefill_chunk_size() == 512


def test_an_explicit_argument_beats_the_env(monkeypatch):
    monkeypatch.setenv("VQLAB_PREFILL_CHUNK", "512")
    assert prefill_chunk_size(64) == 64


def test_a_blank_env_is_not_a_setting(monkeypatch):
    monkeypatch.setenv("VQLAB_PREFILL_CHUNK", "  ")
    assert prefill_chunk_size() == DEFAULT_PREFILL_CHUNK


@pytest.mark.parametrize("bad", ["nope", "3.5"])
def test_a_junk_env_is_an_actionable_error(monkeypatch, bad):
    monkeypatch.setenv("VQLAB_PREFILL_CHUNK", bad)
    with pytest.raises(ValueError, match="VQLAB_PREFILL_CHUNK"):
        prefill_chunk_size()


@pytest.mark.parametrize("bad", [0, -1])
def test_a_nonpositive_chunk_is_rejected(bad):
    with pytest.raises(ValueError, match=">= 1"):
        prefill_chunk_size(bad)


def test_the_env_reaches_the_loop(monkeypatch):
    monkeypatch.setenv("VQLAB_PREFILL_CHUNK", "3")
    ref, _ = run(prompt_of(11), 10_000)
    got, head = run(prompt_of(11), None)
    assert got == ref
    # 11 prompt tokens, 10 seeded positions, chunk 3 -> 4 advance calls.
    assert len(head.seed_hidden) == 4


# --- the identical-tokens gate ----------------------------------------------

# 8 is the width; 5 is short of it, 8 is an exact multiple of the SEEDED
# length, 16 an exact multiple of the prompt, 17/23 ragged tails.
@pytest.mark.parametrize("n_prompt", [1, 2, 3, 5, 8, 9, 16, 17, 23])
@pytest.mark.parametrize("step", [1, 2, 3, 8, 10_000])
def test_tokens_are_identical_at_every_chunk_width(n_prompt, step):
    prompt = prompt_of(n_prompt)
    ref, _ = run(prompt, 10_000)
    got, _ = run(prompt, step)
    assert got == ref


@pytest.mark.parametrize("n_prompt", [1, 2, 5, 8, 9, 16, 17, 23])
@pytest.mark.parametrize("step", [1, 2, 3, 8])
def test_the_head_sees_the_same_seed_stream_as_an_unchunked_prefill(
        n_prompt, step):
    prompt = prompt_of(n_prompt)
    _, ref_head = run(prompt, 10_000)
    _, head = run(prompt, step)
    ref_h, ref_ids = ref_head.seed_stream()
    got_h, got_ids = head.seed_stream()
    assert got_ids == ref_ids
    assert got_h.shape == ref_h.shape
    assert mx.allclose(got_h, ref_h).item()


@pytest.mark.parametrize("n_prompt", [1, 2, 5, 8, 9, 17])
def test_the_seed_covers_positions_0_to_P_minus_2_exactly(n_prompt):
    """The head's input at position j is (h_j, x_{j+1}). Position P-1 needs
    x_P -- the first SAMPLED token -- so it is the bootstrap draft, not part
    of the seed. The seed is therefore ids[1:] and nothing else."""
    prompt = prompt_of(n_prompt)
    _, head = run(prompt, 3)
    _, ids = head.seed_stream()
    assert ids == [int(t) for t in prompt[0].tolist()][1:]


def test_legacy_alignment_does_not_seed_at_all():
    _, head = run(prompt_of(17), 4, head=RecordingHead("biased"))
    legacy = RecordingHead("biased")
    [r.token for r in mtp_stream_generate(
        toy.ToyModel(), Tok(), prompt_of(17), legacy, max_tokens=8,
        prefill_step_size=4, align="legacy")]
    assert head.seed_hidden and not legacy.seed_hidden
