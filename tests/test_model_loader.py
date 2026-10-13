"""family/model_loader.py dispatch logic, CPU only, loads nothing.

Moved here from model_loader._selftest (untangle step 14; the plan's
core/runtime_load row): it checks the module's own functions, so it is a
unit test, not a pipeline stage.
"""
import pytest

from vqlab.family import model_loader as ML
from vqlab.family.registry import FAMILY


@pytest.mark.parametrize("family,want", [
    ("qwen4_exp", "mlx_lm"),
    ("qwen3_5", "mlx_lm"),          # no `runtime` field
    ("no_such_family", "mlx_lm"),   # unknown family -> the mlx_lm default
])
def test_runtime_for(family, want):
    assert ML.runtime_for(family) == want


def test_declared_mlx_vlm_family(monkeypatch):
    # glm5_next declares mlx_vlm; it loads through mlx_lm only when
    # Knurlogic's arch files serve its model_type. The old _selftest asserted
    # mlx_vlm unconditionally, so it FAILED on any box where they do.
    from vqlab.core.box import arch_finder
    monkeypatch.setattr(arch_finder, "enabled", lambda: False)
    assert ML.runtime_for("glm5_next") == "mlx_vlm"
    monkeypatch.setattr(arch_finder, "enabled", lambda: True)
    monkeypatch.setattr(arch_finder, "available",
                        lambda: {FAMILY["glm5_next"]["model_type"]})
    assert ML.runtime_for("glm5_next") == "mlx_lm"


def test_family_for_model_type():
    assert ML.family_for_model_type("glm5_next") == "glm5_next"
    assert ML.family_for_model_type("qwen3_5_moe") is None      # unclaimed


def test_bad_runtime_value_fails(monkeypatch):
    # both directions (III.5): a bad runtime value must raise, not default
    monkeypatch.setitem(FAMILY["glm5_next"], "runtime", "torch")
    with pytest.raises(ValueError):
        ML.runtime_for("glm5_next")


def test_load_installs_the_vendored_arch_whatever_the_invocation(monkeypatch):
    # 2026-10-13: a copy of stream_convert.py run as a script never installed
    # the finder, loaded stock qwen3_5 and wrote bf16 norms where the CLI
    # wrote float32. The loader installs it itself now.
    from vqlab.core.box import arch_finder
    calls = []
    monkeypatch.setattr(arch_finder, "install_arch_finder",
                        lambda: calls.append(1) or True)
    monkeypatch.setattr("mlx_lm.utils.load_model",
                        lambda path, **kw: ("model", {}), raising=False)
    pytest.importorskip("mlx_lm")
    ML._load_on_current_stream("qwen3_5", "x", "mlx_lm", True)
    assert calls == [1]


def test_require_vendored_arch_refuses_a_silent_stock_build(monkeypatch):
    from vqlab.core.box import arch_finder
    monkeypatch.delenv("VQLAB_VENDORED_ARCH", raising=False)
    monkeypatch.setattr(arch_finder, "_register", lambda: None)   # no knurlogic
    with pytest.raises(SystemExit, match="knurlogic is not importable"):
        ML.require_vendored_arch()
    monkeypatch.setenv("VQLAB_VENDORED_ARCH", "0")                # asked for
    ML.require_vendored_arch()
