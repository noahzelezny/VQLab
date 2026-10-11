"""Every reader of queue records resolves the dir the same way
(config.queues_dir: $VQLAB_QUEUE_DIR, else <records>/queues)."""
import json

from vqlab import config
from vqlab.agents.mcp import tools_runs
from vqlab.ops import queues as ops_queues
from vqlab.queue import files, report

NAME = "20261010-000000-fake"


def _fake(d):
    q = d / NAME
    q.mkdir(parents=True)
    (q / "state.json").write_text(json.dumps({
        "status": "passed", "pid": None, "commit": "0" * 40,
        "steps": [{"name": "s", "status": "pass", "seconds": 1}]}))
    return q


def _all_see(d, capsys):
    assert config.queues_dir() == d
    assert ops_queues._queues_dir() == d
    assert files.queues_dir() == d
    st = ops_queues.queue_status()
    assert st["queues_dir"] == str(d) and len(st["queues"]) == 1
    assert NAME in json.dumps(tools_runs.t_queue_status(), default=str)
    assert report.list_queues() == 0
    assert NAME in capsys.readouterr().out


def test_env_queue_dir(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VQLAB_QUEUE_DIR", str(tmp_path / "q"))
    _fake(tmp_path / "q")
    _all_see(tmp_path / "q", capsys)


def test_records_default(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("VQLAB_QUEUE_DIR", raising=False)
    monkeypatch.setenv("VQLAB_RECORDS", str(tmp_path / "rec"))
    _fake(tmp_path / "rec" / "queues")
    _all_see(tmp_path / "rec" / "queues", capsys)
