"""The decode loop, against a toy model whose logits depend on the whole
committed prefix.

Greedy speculative decoding must reproduce plain single-token greedy decoding
EXACTLY here. That gate is legitimate on this model and not on a real one: the
toy has no chunked-vs-single-token kernel to disagree with itself, so any
difference is a rollback or bookkeeping bug rather than numerics (see the
chunk control in vqlab/mtp/bench.py for the real-model version).
"""
import mlx.core as mx
import pytest

import toy_family as toy
from vqlab.mtp import mtp_stream_generate, resolve
from vqlab.mtp.sampling import make_distribution


class Tok:
    """Just enough tokenizer for the loop: no detokenizer, no eos."""
    eos_token_ids: set = set()

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


@pytest.fixture(autouse=True)
def family():
    toy.install()
    yield
    toy.remove()


PROMPT = mx.array([[1, 2, 3]])


def plain_greedy(model, n, temp=0.0):
    dist = make_distribution(temp)
    cache = model.make_cache()
    row = model(PROMPT, cache=cache)[:, -1]
    t = mx.argmax(row, axis=-1) if dist is None else dist(row).sample()
    out = []
    for _ in range(n):
        out.append(int(t.item()))
        row = model(t[None], cache=cache)[:, -1]
        t = mx.argmax(row, axis=-1) if dist is None else dist(row).sample()
    return out


def run(model, head, n, **kw):
    return [r.token for r in mtp_stream_generate(
        model, Tok(), PROMPT, head, max_tokens=n, **kw)]


@pytest.mark.parametrize("mode", ["stubborn", "biased"])
def test_greedy_matches_single_token_decoding(mode):
    model = toy.ToyModel()
    ref = plain_greedy(model, 16)
    assert run(model, toy.ToyHead(mode), 16) == ref


def test_every_step_rejects_when_the_draft_is_always_wrong():
    """The rollback-and-replay path, exercised on every single step."""
    model = toy.ToyModel()
    ref = plain_greedy(model, 16)
    head = toy.ToyHead("antioracle", oracle=ref)
    last = None
    out = []
    for last in mtp_stream_generate(model, Tok(), PROMPT, head, max_tokens=16):
        out.append(last.token)
    assert out == ref, "rollback-and-replay must still emit the reference"
    assert last.acceptance == 0.0
    assert last.steps == 8


def test_every_step_accepts_when_the_draft_is_right():
    """The fast path, and the proof that acceptance is doing real work: an
    oracle head fed the trunk's own choices is accepted every time, and the
    output is still the single-token reference."""
    model = toy.ToyModel()
    ref = plain_greedy(model, 16)
    head = toy.ToyHead("oracle", oracle=ref)
    out = run(model, head, 16)
    assert out == ref
    model2 = toy.ToyModel()
    last = None
    for last in mtp_stream_generate(model2, Tok(), PROMPT,
                                    toy.ToyHead("oracle", oracle=ref),
                                    max_tokens=16):
        pass
    assert last.acceptance == 1.0
    assert last.steps == 8


def test_accepted_tokens_are_flagged_as_drafted():
    model = toy.ToyModel()
    ref = plain_greedy(model, 8)
    flags = [r.from_draft for r in mtp_stream_generate(
        model, Tok(), PROMPT, toy.ToyHead("oracle", oracle=ref), max_tokens=8)]
    assert flags == [False, True] * 4


def test_odd_max_tokens_stops_mid_pair():
    model = toy.ToyModel()
    ref = plain_greedy(model, 7)
    out = run(model, toy.ToyHead("biased"), 7)
    assert out == ref and len(out) == 7


def test_eos_stops_generation():
    model = toy.ToyModel()
    ref = plain_greedy(model, 24)
    stop = ref[5]

    class EosTok(Tok):
        eos_token_ids = {stop}

    out = [r.token for r in mtp_stream_generate(
        model, EosTok(), PROMPT, toy.ToyHead("biased"), max_tokens=24)]
    assert out == ref[: ref.index(stop) + 1]
    assert out[-1] == stop


def test_capture_is_removed_after_generation():
    """The wrap is a context manager, not a monkeypatch: the trunk is left
    exactly as it was found."""
    model = toy.ToyModel()
    before = model.model.hyper_connection_mixer
    run(model, toy.ToyHead("biased"), 4)
    assert model.model.hyper_connection_mixer is before


def test_capture_is_removed_even_when_generation_raises():
    model = toy.ToyModel()
    before = model.model.hyper_connection_mixer

    class Boom(toy.ToyHead):
        def draft_logits(self, *a, **k):
            raise ZeroDivisionError("boom")

    with pytest.raises(ZeroDivisionError):
        run(model, Boom(), 4)
    assert model.model.hyper_connection_mixer is before


def test_registry_resolves_the_family_from_model_type():
    spec = resolve(toy.ToyModel())
    assert spec.name == toy.FAMILY
    assert spec.head_cls() is toy.ToyHead
    assert isinstance(spec.make_draft_cache(
        spec.arch_module(toy.ToyModel())), toy.ToyDraftCache)


def test_unknown_family_is_an_actionable_error():
    class Alien(toy.ToyModel):
        model_type = "not_registered"

    with pytest.raises(KeyError, match="no MTP family registered"):
        resolve(Alien())


def test_sampled_output_distribution_matches_plain_sampling():
    """The claim that makes temperature safe: with a deliberately bad draft
    head, the speculative loop's output distribution is still the one plain
    sampling produces. Compared as the joint distribution over the first two
    tokens, which is where an acceptance bug would show first."""
    n, temp = 3000, 1.0
    model = toy.ToyModel()

    def hist(sampler):
        mx.random.seed(20260830)
        counts = {}
        for _ in range(n):
            pair = tuple(sampler())
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    ref = hist(lambda: plain_greedy(model, 2, temp=temp))
    got = hist(lambda: run(model, toy.ToyHead("stubborn"), 2, temp=temp))

    keys = set(ref) | set(got)
    tv = 0.5 * sum(abs(ref.get(k, 0) - got.get(k, 0)) for k in keys) / n
    assert tv < 0.05, f"total variation {tv:.4f} between speculative and plain"
    assert len(keys) > 3, "degenerate distribution: the test proves nothing"


# --------------------------------------------------------------- alignment
# qwen4_exp takes the head's ROTARY POSITIONS from its cache offset
# (Attention.__call__: `offset = cache.offset`, then `rope(_positions(...))`),
# so the offset the head sees at each draft is the thing that has to be right.
# The toy head records it.

def test_committed_alignment_feeds_true_sequence_positions():
    model = toy.ToyModel()
    head = toy.ToyHead("biased")
    run(model, head, 12)
    P = PROMPT.shape[1]
    # bootstrap drafts at position P-1; each step then advances two positions
    # over the pair that committed, so the next draft starts at P, P+2, ...
    expected = [P - 1] + [P + 2 * k for k in range(12 // 2 - 1)]
    assert head.draft_offsets == expected


def test_legacy_alignment_feeds_wrong_positions():
    """The bug, asserted as a bug so the fix cannot silently regress.

    Driven by an oracle head so every step accepts and the sequence is
    deterministic: the head advances one position per step while two tokens
    commit, so it sees 0,1,2,... where the true positions are P-1, P+1, P+3.
    The error is not a constant shift the rope could tolerate — it grows with
    generation length."""
    model = toy.ToyModel()
    ref = plain_greedy(model, 12)
    head = toy.ToyHead("oracle", oracle=ref)
    assert run(model, head, 12, align="legacy") == ref
    P = PROMPT.shape[1]
    assert head.draft_offsets == list(range(12 // 2))
    true_positions = [P - 1 + 2 * k for k in range(12 // 2)]
    drift = [t - o for t, o in zip(true_positions, head.draft_offsets)]
    assert drift == sorted(drift) and drift[-1] > drift[0]


def test_legacy_rollback_starves_the_head_cache():
    """A second legacy defect the alignment fix removes outright: the head's
    cache is rolled back on every rejection, so its history is repeatedly
    discarded rather than accumulated. With a head that never lands, it never
    advances past position 0 across the whole generation.

    align="committed" cannot do this — it feeds the head only committed
    tokens, after verification, so there is nothing to roll back."""
    model = toy.ToyModel()
    ref = plain_greedy(model, 12)
    head = toy.ToyHead("antioracle", oracle=ref)
    run(model, head, 12, align="legacy")
    assert head.draft_offsets == [0] * (12 // 2)

    aligned = toy.ToyHead("antioracle", oracle=plain_greedy(toy.ToyModel(), 12))
    run(toy.ToyModel(), aligned, 12)
    assert aligned.draft_offsets == [2, 3, 5, 7, 9, 11]


@pytest.mark.parametrize("mode", ["stubborn", "biased"])
def test_both_schemes_agree_on_output(mode):
    """Alignment is an ACCEPTANCE knob, not a correctness one: the trunk
    verifies every drafted token either way, so both schemes must emit the
    single-token reference."""
    model = toy.ToyModel()
    ref = plain_greedy(model, 16)
    assert run(model, toy.ToyHead(mode), 16) == ref
    assert run(model, toy.ToyHead(mode), 16, align="legacy") == ref


def test_align_rejects_an_unknown_scheme():
    with pytest.raises(ValueError, match="align must be"):
        run(toy.ToyModel(), toy.ToyHead("biased"), 4, align="nonsense")
