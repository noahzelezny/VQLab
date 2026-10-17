"""config.require_free: refuses a writer before it starts when the output
volume cannot hold its estimate; the override env var skips it."""
import pytest

from vqlab import config


def test_refuses_when_too_big(tmp_path):
    with pytest.raises(SystemExit, match="REFUSED"):
        config.require_free(tmp_path / "new" / "dir", 2**60, "test")


def test_passes_small_and_override(tmp_path, monkeypatch):
    config.require_free(tmp_path / "out", 1024, "test", margin_gib=0)
    monkeypatch.setenv("VQLAB_SKIP_DISK_CHECK", "1")
    config.require_free(tmp_path, 2**60, "test")


def test_free_when_short_reports_the_room_it_found(tmp_path, monkeypatch):
    free = config.free_when_short(tmp_path, 2**60)
    assert isinstance(free, int) and 0 < free < 2**60
    assert config.free_when_short(tmp_path, 1024, margin_gib=0) is None
    monkeypatch.setenv("VQLAB_SKIP_DISK_CHECK", "1")
    assert config.free_when_short(tmp_path, 2**60) is None
