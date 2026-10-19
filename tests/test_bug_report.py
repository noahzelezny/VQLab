"""agents/bug_report: a prefilled GitHub issue that carries no paths."""
import urllib.parse

from vqlab.agents import bug_report
from vqlab.records import runlog


def test_the_last_failure_names_its_command_and_never_its_arguments(monkeypatch):
    monkeypatch.setattr(runlog, "runs", lambda: [
        {"cmd": "fit-moe", "time": "t1", "argv": ["--src", "/private/x"], "end": {"rc": 1}},
        {"cmd": "doctor", "time": "t2", "argv": [], "end": {"rc": 0}},
        {"cmd": "gui", "time": "t3", "argv": [], "end": None}])
    fail = bug_report.last_failure()
    assert fail == {"cmd": "fit-moe", "time": "t1", "rc": "1"}
    assert "/private/x" not in bug_report.body()


def test_the_body_carries_the_versions(monkeypatch):
    monkeypatch.setattr(runlog, "runs", list)
    text = bug_report.body("it broke")
    assert "it broke" in text and "vqlab: " in text and "mlx: " in text
    assert "Last failed command" not in text


def test_the_link_opens_a_new_bug_on_the_project_tracker_and_fits(monkeypatch):
    url = bug_report.new_issue_url("bug: x", "y" * 50_000)
    assert url.startswith(bug_report.issues_url().rstrip("/") + "/new?")
    assert len(url) <= bug_report.MAX_URL
    q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    assert q["title"] == ["bug: x"] and q["labels"] == ["bug"]
