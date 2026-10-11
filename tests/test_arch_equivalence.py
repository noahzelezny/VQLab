"""arch_equivalence.json: a cache whose ONLY stamp difference is a listed
(model_type, arch sha pair) is accepted; every other difference still
refuses. Synthetic stamps; CPU only."""
import json

import pytest

from vqlab.family import version_stamp as N

REG = json.loads(N.ARCH_EQUIVALENCE.read_text())
E = REG["entries"][0]
RUN = {"mlx": "0.32.3", "mlx_lm": "0.32.0", "model_type": E["model_type"],
       "arch_sha256": E["new_sha256"], "scorer_variant": None}
OLD = {**RUN, "arch_sha256": E["old_sha256"]}


def test_registry_entries_are_complete_and_content_addressed():
    for e in REG["entries"]:
        assert {"model_type", "old_sha256", "new_sha256", "evidence", "date"} <= set(e)
        assert len(e["old_sha256"]) == len(e["new_sha256"]) == 64
        assert e["old_sha256"] != e["new_sha256"]


def test_listed_pair_is_accepted_with_a_note(capsys):
    rec = N.check_cache({"numerics": OLD}, RUN)
    assert rec["status"] == "arch-equivalent"
    assert rec["equivalence"]["old_sha256"] == E["old_sha256"]
    assert "arch_equivalence.json" in capsys.readouterr().out
    # content-addressed: either direction
    assert N.check_cache({"numerics": RUN}, OLD)["status"] == "arch-equivalent"


def test_unlisted_sha_refuses():
    with pytest.raises(SystemExit, match="arch_sha256"):
        N.check_cache({"numerics": {**RUN, "arch_sha256": "0" * 64}}, RUN)


def test_listed_pair_under_another_model_type_refuses():
    with pytest.raises(SystemExit, match="arch_sha256"):
        N.check_cache({"numerics": {**OLD, "model_type": "qwen4_exp"}},
                      {**RUN, "model_type": "qwen4_exp"})


def test_listed_pair_plus_any_other_difference_refuses():
    for k, v in (("mlx", "0.32.0.dev"), ("mlx_lm", "0.31.9"), ("scorer_variant", "x")):
        with pytest.raises(SystemExit, match=k):
            N.check_cache({"numerics": {**OLD, k: v}}, RUN)


def test_pairing_accepts_listed_pair_and_refuses_unlisted():
    def side(build):
        return {"kl_positions": 4, "numerics": dict(build),
                "kl_cache": {"tokens_sha256": "t", "teacher_head_sha256": "h", "teacher_bytes": 9}}
    assert N.pairing_problems(side(OLD), side(RUN)) == []
    probs = N.pairing_problems(side({**OLD, "arch_sha256": "f" * 64}), side(RUN))
    assert any("arch_sha256" in p for p in probs)
