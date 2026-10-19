"""agents/updates: a newer vqlab on PyPI, asked at most once a day."""
import pytest

import vqlab
from vqlab.agents import updates


@pytest.fixture(autouse=True)
def online(monkeypatch):
    monkeypatch.delenv("VQLAB_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    (updates.config.records() / "update-check.json").unlink(missing_ok=True)


def test_versions_order_by_number_not_text():
    assert updates.version_key("0.1.10") > updates.version_key("0.1.9")
    assert updates.version_key("1.0") > updates.version_key("0.9.9")


def test_a_newer_release_is_an_update_and_an_equal_one_is_not():
    newer = updates.release_doc(ask=lambda: "99.0.0")
    assert newer["update"] and newer["latest"] == "99.0.0" and newer["command"] == updates.COMMAND
    (updates.config.records() / "update-check.json").unlink()
    assert not updates.release_doc(ask=lambda: vqlab.__version__)["update"]


def test_pypi_is_asked_once_a_day_and_a_failure_is_not_cached():
    calls = []

    def ask():
        calls.append(1)
        return "9.9.9"

    def down():
        raise OSError("no network")

    assert updates.latest(down, now=0) is None
    assert updates.latest(ask, now=0) == "9.9.9"
    assert updates.latest(ask, now=updates.MAX_AGE_S - 1) == "9.9.9"
    assert len(calls) == 1
    assert updates.latest(ask, now=updates.MAX_AGE_S + 1) == "9.9.9" and len(calls) == 2


def test_offline_never_asks(monkeypatch):
    monkeypatch.setenv("VQLAB_OFFLINE", "1")
    assert updates.latest(lambda: pytest.fail("asked PyPI while offline")) is None
