"""The MTP pack recipe's one home (mtp.registry: the default recipe and the
sidecar's name) and the wiring a queue's mtp-pack reads from its probe
(plan/mtp_probe_qwen35.chosen_wiring)."""
import json

import pytest

from vqlab.mtp import registry
from vqlab.plan.mtp_probe_qwen35 import chosen_wiring


def test_a_sidecar_is_named_for_its_recipe():
    spec = registry.FAMILIES["qwen3_5_moe"]
    assert spec.sidecar_for() == "mtp-head-q6.safetensors"
    assert spec.sidecar_for(8) == "mtp-head-q8.safetensors"
    assert spec.sidecar_for(8, 3) == "mtp-head-e3q8.safetensors"


def _probe(tmp_path, sidecars):
    f = tmp_path / "probe.json"
    f.write_text(json.dumps({"sidecars": {n: {"best": b} for n, b in sidecars.items()}}))
    return f


def test_the_probe_s_best_wiring_is_read(tmp_path):
    f = _probe(tmp_path, {"mtp-head-q6.safetensors": "eh/post_norm"})
    assert chosen_wiring(f) == ("eh", "post_norm")


def test_a_probe_of_several_sidecars_chooses_nothing(tmp_path):
    f = _probe(tmp_path, {"a": "eh/pre_norm", "b": "eh/post_norm"})
    with pytest.raises(SystemExit, match="exactly one"):
        chosen_wiring(f)
