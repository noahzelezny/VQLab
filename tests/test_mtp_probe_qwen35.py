"""plan/mtp_probe_qwen35.chosen_wiring: the wiring a queue's mtp-pack reads
from its probe record."""
import json

import pytest

from vqlab.plan.mtp_probe_qwen35 import chosen_wiring


def _probe(tmp_path, sidecars):
    f = tmp_path / "probe.json"
    f.write_text(json.dumps({"sidecars": {n: {"best": b} for n, b in sidecars.items()}}))
    return f


def test_the_probe_s_best_wiring_is_read(tmp_path):
    f = _probe(tmp_path, {"mtp-head-q6.safetensors": "eh/post_norm"})
    assert chosen_wiring(f) == ("eh", "post_norm")


def test_a_probe_of_several_sidecars_chooses_nothing(tmp_path):
    f = _probe(tmp_path, {"a": "eh/pre_norm", "b": "eh/post_norm"})
    with pytest.raises(ValueError, match="exactly one"):
        chosen_wiring(f)


def test_a_probe_wiring_mtp_pack_does_not_take_is_refused(tmp_path):
    f = _probe(tmp_path, {"a": "eh/post-norm"})
    with pytest.raises(ValueError, match="not one mtp-pack takes"):
        chosen_wiring(f)
