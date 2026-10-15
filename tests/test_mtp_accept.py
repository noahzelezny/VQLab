"""mtp-accept's pooled acceptance: a head with no draft step accepted nothing."""
from vqlab.bench.mtp_accept import pooled_acceptance


def test_pooled_acceptance_pools_and_survives_zero_steps():
    rows = [{"head": "h", "align": "a", "accepted": 3, "steps": 4},
            {"head": "h", "align": "a", "accepted": 1, "steps": 4},
            {"head": "z", "align": "a", "accepted": 0, "steps": 0}]
    assert pooled_acceptance(rows, ["h", "z"], ["a"]) == {"h/a": 0.5, "z/a": 0.0}
