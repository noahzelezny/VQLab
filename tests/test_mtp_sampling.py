"""Rejection sampling must preserve the target distribution for ANY draft.

That is the whole quality argument for speculative decoding at temperature —
the greedy loop gets it from `draft == argmax(p)`, and this is the general
case. It is checked two ways: the closed-form outcome distribution (exact, no
sampling), and the sampler as actually implemented (empirical).
"""
import mlx.core as mx
import pytest

from vqlab.mtp.sampling import (
    acceptance_profile,
    make_distribution,
    rejection_correct,
)


def _dirichlet(rows, vocab, seed):
    mx.random.seed(seed)
    return mx.softmax(mx.random.normal((rows, vocab)) * 2.0, axis=-1)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_outcome_distribution_is_exactly_p(seed):
    """Closed form: accept-or-residual is p, whatever q is."""
    p = _dirichlet(8, 32, seed)
    q = _dirichlet(8, 32, seed + 100)
    accept, out = acceptance_profile(p, q)
    assert float(mx.abs(out - p).max().item()) < 1e-6
    assert float(mx.abs(out.sum(-1) - 1).max().item()) < 1e-5
    assert bool(mx.all((accept >= 0) & (accept <= 1)).item())


def test_acceptance_is_one_iff_draft_matches_target():
    p = _dirichlet(4, 16, 7)
    accept, out = acceptance_profile(p, p)
    assert float(mx.abs(accept - 1.0).max().item()) < 1e-5
    assert float(mx.abs(out - p).max().item()) < 1e-6


def test_residual_correction_is_not_resampling_from_p():
    """Guard against the tempting simplification. Resampling the replacement
    from p instead of the residual biases the output; assert the residual is
    genuinely a different distribution when q is a bad draft."""
    p = mx.array([[0.7, 0.2, 0.1]])
    q = mx.array([[0.1, 0.1, 0.8]])
    residual = mx.maximum(p - q, 0.0)
    residual = residual / residual.sum(-1, keepdims=True)
    assert float(mx.abs(residual - p).max().item()) > 0.1
    _, out = acceptance_profile(p, q)
    assert float(mx.abs(out - p).max().item()) < 1e-6


def test_empirical_draw_matches_p():
    """The implementation, not just the algebra: draw from q, correct, and
    compare the histogram against p."""
    mx.random.seed(1234)
    vocab, n = 6, 40000
    p = mx.array([[0.30, 0.25, 0.20, 0.15, 0.07, 0.03]])
    q = mx.array([[0.05, 0.05, 0.10, 0.20, 0.30, 0.30]])   # a bad draft
    pb = mx.broadcast_to(p, (n, vocab))
    qb = mx.broadcast_to(q, (n, vocab))
    draft = mx.random.categorical(mx.log(qb))
    accepted, out = rejection_correct(pb, qb, draft)
    emp = mx.array([float((out == v).sum().item()) for v in range(vocab)]) / n
    tv = 0.5 * float(mx.abs(emp - p[0]).sum().item())
    assert tv < 0.01, f"total variation {tv:.4f} from p"
    # And the measured acceptance rate matches the closed-form one.
    expected, _ = acceptance_profile(p, q)
    got = float(accepted.mean().item())
    assert abs(got - float(expected.item())) < 0.01


def test_make_distribution_is_greedy_at_zero_temperature():
    assert make_distribution(0.0) is None


def test_distribution_is_normalised_and_respects_top_k():
    dist = make_distribution(temp=0.8, top_k=3)
    logits = mx.array([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0]])
    d = dist(logits)
    assert abs(float(d.probs.sum().item()) - 1.0) < 1e-5
    assert float(d.probs[0, 3:].sum().item()) < 1e-6
    assert int(d.argmax().item()) == 0


def test_temperature_sharpens_the_distribution():
    logits = mx.array([[2.0, 1.0, 0.0]])
    hot = make_distribution(temp=2.0)(logits).probs
    cold = make_distribution(temp=0.2)(logits).probs
    assert float(cold[0, 0].item()) > float(hot[0, 0].item())
