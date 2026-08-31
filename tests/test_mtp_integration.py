"""End-to-end against a real artifact and sidecar. Skipped unless both are
pointed at by the environment:

    export VQLAB_MTP_MODEL="/Volumes/Thunderbay SSD/Exo Models/TheDrainFlorist--Qwen3.8-Flash-Next-VQ-2.1bpw"
    export VQLAB_MTP_SIDECAR="/Volumes/Thunderbay SSD/Exo Models/mtp-head-q6.safetensors"
    pytest tests/test_mtp_integration.py

MEMORY: this loads the whole trunk (45 GiB at the 2.1bpw rung) plus a 2.12 GiB
head. Do not run it alongside anything else large.

Note the correctness gate. Bit-identical output against single-token decoding
is NOT achievable on MLX and demanding it is a false gate: the chunked and
single-token kernels disagree at genuine near-ties (measured top-2 logit gaps
of 0.25 and 0.00 against a median of 3.625). The gate asserted here is the
correct one — divergence confined to near-ties, measured by the same chunk
control the baseline itself fails.
"""
import os

import pytest

pytestmark = pytest.mark.integration

MODEL = os.environ.get("VQLAB_MTP_MODEL")
SIDECAR = os.environ.get("VQLAB_MTP_SIDECAR")

pytest.importorskip("mlx_lm")

if not MODEL:
    pytest.skip("set VQLAB_MTP_MODEL (and VQLAB_MTP_SIDECAR)",
                allow_module_level=True)


@pytest.fixture(scope="module")
def loaded():
    from mlx_lm.utils import load

    from vqlab.mtp import load_mtp_head
    model, tok = load(MODEL, lazy=False, trust_remote_code=True)
    head, spec = load_mtp_head(model, sidecar=SIDECAR, model_path=MODEL)
    return model, tok, head, spec


def _prompt(tok):
    return tok.apply_chat_template(
        [{"role": "user", "content": "Name three primary colors."}],
        add_generation_prompt=True)


def test_cache_semantics_claim_holds_for_this_family(loaded):
    """The registry claims cache_semantics='reassign' for qwen4_exp — i.e.
    that non-copying snapshots are safe because slots are replaced, never
    written into. Measure it rather than assume it."""
    import mlx.core as mx

    from vqlab.mtp.caches import check_snapshot_semantics
    model, tok, _, spec = loaded
    cache = model.make_cache()
    ids = mx.array([_prompt(tok)])
    mx.eval(model(ids, cache=cache))
    last = ids[:, -1:]
    ok = check_snapshot_semantics(cache, lambda: mx.eval(
        model(last, cache=cache)))
    assert ok == (spec.cache_semantics == "reassign")


def test_greedy_divergence_is_confined_to_near_ties(loaded):
    from vqlab.mtp.bench import benchmark
    model, tok, head, _ = loaded
    rec = benchmark(model, tok, _prompt(tok), head, tokens=64, warmup=4)
    assert rec["acceptance"] > 0.3, rec
    assert rec["speedup"] > 1.0, rec
    gaps = rec["chunk_control_gaps"]
    median = rec["chunk_control_median_gap"]
    assert all(g < 0.25 * median for g in gaps), (
        f"divergence at gaps {gaps} against median {median}: that is not a "
        f"near-tie, it is a rollback bug")


def test_sampling_runs_and_produces_text(loaded):
    from vqlab.mtp import mtp_generate
    model, tok, head, _ = loaded
    text = mtp_generate(model, tok, _prompt(tok), head, max_tokens=32,
                        temp=0.7, top_p=0.9)
    assert text.strip()
