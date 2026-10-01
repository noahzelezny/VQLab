"""speed-pair-knurlogic expects a bundled runtime only when the arm ships a model.py."""
from vqlab.bench.speed_pair_knurlogic import ships_runtime


def test_ships_runtime(tmp_path):
    vq = tmp_path / "vq-build"
    vq.mkdir()
    (vq / "config.json").write_text("{}")
    (vq / "model.py").write_text("# bundled runtime")
    aff = tmp_path / "affine-8bit"
    aff.mkdir()
    (aff / "config.json").write_text("{}")
    assert ships_runtime("vq-build", tmp_path) is True
    assert ships_runtime("affine-8bit", tmp_path) is False
    assert ships_runtime("missing", tmp_path) is None
