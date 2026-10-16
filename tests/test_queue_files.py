"""queue.files: which preflights already have a full run, read from the
queue dirs' state.json `source`."""
import json

from vqlab.queue import files as qf


def _q(root, name, text):
    d = root / name
    d.mkdir()
    (d / "state.json").write_text(text)
    return d


def test_full_run_started_follows_the_sources(tmp_path, monkeypatch):
    monkeypatch.setenv("VQLAB_QUEUE_DIR", str(tmp_path))
    pf = _q(tmp_path, "1-a-preflight", json.dumps({"preflight": True}))
    other = _q(tmp_path, "2-b-preflight", json.dumps({"preflight": True}))
    _q(tmp_path, "3-a", json.dumps({"source": str((pf / "queue.json").resolve())}))
    assert qf.sources() == {str((pf / "queue.json").resolve())}
    assert qf.full_run_started(pf) and not qf.full_run_started(other)
    assert not qf.full_run_started(pf, started=set())   # a precomputed set is used as given


def test_a_half_written_state_is_skipped(tmp_path, monkeypatch):
    monkeypatch.setenv("VQLAB_QUEUE_DIR", str(tmp_path))
    _q(tmp_path, "1-broken", "")
    _q(tmp_path, "2-a", json.dumps({"source": "/x/q.json"}))
    assert qf.sources() == {"/x/q.json"}
