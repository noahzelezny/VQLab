"""Timed commands refuse a busy Mac unless VQLAB_ALLOW_BUSY=1; untimed
commands are never checked."""
import pytest

from vqlab.bench import box_quiet as B


def _busy(monkeypatch, busy):
    monkeypatch.setattr(B, "state", lambda: {"load1": 1.0, "ncpu": 8, "heavy": [],
                                             "lease_holder": None, "busy": busy})


def test_refuses_busy(monkeypatch):
    _busy(monkeypatch, ["GPU lease held by fit"])
    monkeypatch.delenv("VQLAB_ALLOW_BUSY", raising=False)
    with pytest.raises(SystemExit, match="REFUSED"):
        B.guard("decode-timeline")


def test_override_and_untimed(monkeypatch):
    _busy(monkeypatch, ["load"])
    B.guard("mix")                                   # not timed: no check
    monkeypatch.setenv("VQLAB_ALLOW_BUSY", "1")
    B.guard("speed-pair")                            # override


def test_quiet_passes(monkeypatch):
    _busy(monkeypatch, [])
    B.guard("prefill-bench")
