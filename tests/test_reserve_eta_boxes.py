"""Box reservations, queue ETAs, the --on remote path check, and the
[boxes.NAME] teachers override (CPU; ssh is mocked, nothing leaves the box)."""
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
import time

import pytest

from vqlab import _layout  # noqa: F401
from vqlab import config as C
sys.path.insert(0, str(_layout.SRC / "vqlab" / "agents"))
import mcp_server as ms  # noqa: E402
import queue_eta  # noqa: E402
import reserve as R  # noqa: E402
import run_queue as rq  # noqa: E402


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """A config with an m4 box (teachers override) and scratch under tmp."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    f = tmp_path / "config.toml"
    f.write_text(f'[paths]\nscratch = "{scratch}"\nteachers = "{tmp_path}/hdd-teachers"\n'
                 f'[boxes.m4]\nssh = "u@m4"\nrepo = "{tmp_path}/clone"\npython = "/py"\n'
                 f'config = "{tmp_path}/m4.toml"\nqueue_dir = "{tmp_path}/q-m4"\n'
                 f'teachers = "{tmp_path}/m4-local-teachers"\n')
    for k in ("VQLAB_BOX", "VQLAB_TEACHERS_DIR", "VQLAB_RESERVATIONS", "VQLAB_SCRATCH", "VQLAB_WHO"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("VQLAB_CONFIG", str(f))
    monkeypatch.setattr(C, "HOME", tmp_path / "home")
    C._file_paths.cache_clear()
    C._file_table.cache_clear()
    yield tmp_path
    C._file_paths.cache_clear()
    C._file_table.cache_clear()


# ------------------------------------------------------------ reservations
def test_parse_until():
    now = dt.datetime(2026, 10, 3, 12, 0, tzinfo=dt.timezone.utc)
    assert R.parse_until("+2h30m", now) == now + dt.timedelta(hours=2.5)
    assert R.parse_until("+1d", now) == now + dt.timedelta(days=1)
    assert R.parse_until("2026-10-04T01:00:00+00:00", now).hour == 1
    t = R.parse_until("00:01", now)
    assert now < t <= now + dt.timedelta(days=1)
    with pytest.raises(ValueError):
        R.parse_until("+3x", now)


def test_reserve_list_release_and_refusal(cfg, monkeypatch, capsys):
    monkeypatch.setenv("VQLAB_WHO", "noah")
    R.main(["m4", "--for", "knurlogic", "--until", "+2h", "--note", "vision bench"])
    store = cfg / "scratch" / R.FILE
    assert store.exists(), "reservations go to the shared scratch root"
    assert R.get("m4")["for"] == "knurlogic"
    why = R.refusal("m4")
    assert why and "knurlogic" in why and "--override-reservation" in why
    assert R.refusal("m4", override=True) is None
    monkeypatch.setenv("VQLAB_WHO", "knurlogic")
    assert R.refusal("m4") is None                      # your own reservation
    monkeypatch.setenv("VQLAB_WHO", "noah")
    with pytest.raises(SystemExit):                     # someone else's: --force needed
        R.reserve("m4", "noah", "+1h")
    R.main(["--list"])
    assert "reserved for knurlogic" in capsys.readouterr().out
    R.main(["--release", "m4"])
    assert R.get("m4") is None


def test_expired_reservation_is_ignored(cfg, monkeypatch):
    past = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=5)).isoformat()
    (cfg / "scratch" / R.FILE).write_text(json.dumps(
        {"reservations": [{"box": "m4", "for": "x", "until": past}]}))
    assert R.active() == [] and R.refusal("m4") is None


def test_fallback_when_scratch_unreachable(cfg, monkeypatch, capsys):
    monkeypatch.setenv("VQLAB_RESERVATIONS", str(cfg / "unmounted" / R.FILE))
    r = R.reserve("here", "noah", "+1h")
    assert r["store"] == str(R.local_path())
    assert "visible ONLY" in capsys.readouterr().err
    assert R.active()[0]["local_only"]


def test_gpu_state_lists_reservations(cfg, monkeypatch):
    R.reserve("m4", "knurlogic", "+1h")
    monkeypatch.setattr(ms, "_box_state", lambda n, b: {"stub": True})
    monkeypatch.setattr(ms, "_exo_instances", lambda: [])
    out = ms.t_gpu_state()
    assert [r["box"] for r in out["reservations"]] == ["m4"]
    assert "knurlogic" in out["reservations"][0]["text"]


def test_queue_run_refuses_reserved_box(cfg, monkeypatch):
    monkeypatch.setenv("VQLAB_WHO", "noah")
    monkeypatch.setenv("VQLAB_BOX", "m4")              # "here" is m4
    R.reserve("here", "knurlogic", "+1h")
    called = []
    monkeypatch.setattr(rq, "create", lambda *a, **k: called.append(a))
    with pytest.raises(SystemExit) as e:
        rq.main(["run", str(cfg / "q.json")])
    assert "knurlogic" in str(e.value) and not called
    monkeypatch.delenv("VQLAB_BOX")
    with pytest.raises(SystemExit) as e:                # and --on m4 from the home box
        rq.main(["run", str(cfg / "q.json"), "--on", "m4"])
    assert "knurlogic" in str(e.value)


def test_override_is_recorded(cfg, monkeypatch):
    monkeypatch.setenv("VQLAB_WHO", "noah")
    monkeypatch.setenv("VQLAB_BOX", "m4")
    R.reserve("here", "knurlogic", "+1h")
    qdir = cfg / "qd"
    qdir.mkdir()
    (qdir / "state.json").write_text(json.dumps({"steps": []}))
    monkeypatch.setattr(rq, "create", lambda *a, **k: qdir)
    monkeypatch.setattr(rq, "run", lambda *a, **k: 0)
    assert rq.main(["run", str(cfg / "q.json"), "--override-reservation"]) == 0
    s = json.loads((qdir / "state.json").read_text())
    assert s["reservation_override"]["for"] == "knurlogic"
    assert s["reservation_override"]["overridden_by"] == "noah"


# ------------------------------------------------------------ remote path check
def _fake_ssh(results, seen=None):
    def run(argv, **kw):
        if seen is not None:
            seen.append(argv)
        return subprocess.CompletedProcess(argv, 0, json.dumps(results) + "\n", "")
    return run


def test_remote_check_eio_and_unmounted(monkeypatch):
    b = {"ssh": "u@m4", "python": "/py"}
    seen = []
    monkeypatch.setattr(rq.subprocess, "run", _fake_ssh([
        {"path": "/Volumes/Storage SSD/a", "exists": True, "listed": True,
         "readable": False, "errno": "EIO"},
        {"path": "/Volumes/Storage SSD/b", "exists": True, "readable": False, "errno": "EIO"},
        {"path": "/Volumes/Storage HDD/t", "exists": False, "ancestor": "/Volumes"},
        {"path": "/Volumes/Storage SSD/x/missing", "exists": False,
         "ancestor": "/Volumes/Storage SSD/x"},
        {"path": "/Volumes/Storage SSD/ok", "exists": True, "readable": True}], seen))
    probs = rq.remote_check("m4", b, ["/Volumes/Storage SSD/a"])
    assert probs == [
        "on m4: /Volumes/Storage SSD is listed but unreadable (EIO): remount Storage SSD on m4",
        "on m4: /Volumes/Storage HDD is not mounted (/Volumes/Storage HDD/t does not exist): "
        "mount Storage HDD on m4",
        "on m4: /Volumes/Storage SSD/x/missing does not exist"]
    assert seen[0][:2] == ["ssh", "-o"] and seen[0][-2] == "u@m4"


def test_remote_probe_script_reads_a_file(tmp_path):
    """The probe that runs on the remote, run here: it READS one file."""
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "f.bin").write_bytes(b"x" * 100)
    r = subprocess.run([sys.executable, "-c", rq._REMOTE_PROBE,
                        json.dumps([str(tmp_path / "d"), str(tmp_path / "nope")])],
                       capture_output=True, text=True, check=True)
    a, b = json.loads(r.stdout)
    assert a["readable"] and a["read"].endswith("f.bin")
    assert not b["exists"] and b["ancestor"] == str(tmp_path)


def test_run_on_refuses_before_launch(cfg, monkeypatch):
    q = cfg / "q.json"
    q.write_text(json.dumps({"steps": [{"name": "fit", "cmd": "fit-moe", "preflight": {"skip": "x"},
                                        "args": ["--src", "<teachers>/DS4", "--out", str(cfg / "o" / "x")]}]}))
    seen = []

    def fake(argv, **kw):
        seen.append(argv)
        paths = json.loads(shlex.split(argv[-1])[-1])
        return subprocess.CompletedProcess(argv, 0, json.dumps(
            [{"path": p, "exists": True, "readable": False, "errno": "EIO"} for p in paths]), "")
    monkeypatch.setattr(rq.subprocess, "run", fake)
    monkeypatch.setattr(rq.subprocess, "call", lambda *a, **k: pytest.fail("launched"))
    paths = rq.remote_paths(json.loads(q.read_text()), q, C.boxes()["m4"], "m4")
    assert str(cfg / "m4-local-teachers" / "DS4") in paths     # the box's teachers override
    with pytest.raises(SystemExit) as e:
        rq.run_on("m4", q)
    assert "unreadable (EIO)" in str(e.value) and "nothing launched" in str(e.value)


# ------------------------------------------------------------ teachers override
def test_teachers_override_on_that_box(cfg, monkeypatch):
    assert C.teachers() == cfg / "hdd-teachers"
    monkeypatch.setenv("VQLAB_BOX", "m4")
    assert C.teachers() == cfg / "m4-local-teachers"
    assert C.expand("<teachers>/DS4") == f"{cfg}/m4-local-teachers/DS4"
    assert C.expand("--src=<teachers>/DS4") == f"--src={cfg}/m4-local-teachers/DS4"
    assert C.expand("a<teachers>/x") == "a<teachers>/x"
    monkeypatch.setenv("VQLAB_TEACHERS_DIR", "/env/wins")
    assert str(C.teachers()) == "/env/wins"


def test_config_show_documents_override(cfg, monkeypatch, capsys):
    from vqlab.agents import config_cmd
    monkeypatch.setenv("VQLAB_BOX", "m4")
    config_cmd.show()
    out = capsys.readouterr().out
    assert "m4-local-teachers" in out and "[boxes.m4] teachers" in out


# ------------------------------------------------------------ ETA
def _queue(root, name, steps, recs, preflight=False, status="running"):
    d = root / name
    d.mkdir(parents=True)
    (d / "queue.json").write_text(json.dumps({"steps": steps}))
    (d / "state.json").write_text(json.dumps({"name": name, "status": status, "commit": "abc",
                                              "preflight": preflight, "steps": recs,
                                              "pid": os.getpid()}))
    return d


def test_eta_fit_moe_from_progress(tmp_path):
    st = {"name": "fit", "cmd": "fit-moe", "args": []}
    d = _queue(tmp_path, "q1", [st], [{"name": "fit", "status": "running",
                                       "started": "2026-10-03T00:00:00+00:00"}])
    sd = d / "steps" / "00-fit"
    sd.mkdir(parents=True)
    (sd / "stdout").write_text("    L24 up_proj    relerr 0.1447\n"
                               "[1/7] model-L024.safetensors  (451s)\n"
                               "[2/7] model-L025.safetensors  (900s)\n")
    now = (sd / "stdout").stat().st_mtime + 100
    e = queue_eta.queue_eta(d, tmp_path, now=now)
    assert e["steps"][0]["eta_s"] == 450 * 5 - 100
    assert "2/7 shards" in e["steps"][0]["basis"]
    assert e["eta_s"] == 2150


def test_eta_kl_ladder_cells(tmp_path):
    st = {"name": "kl", "cmd": "kl-ladder",
          "args": ["--cache", "p=/a", "--cache", "c=/b", "--rung", "r1=/x", "--rung", "r2=/y"]}
    d = _queue(tmp_path, "q1", [st], [{"name": "kl", "status": "running",
                                       "started": "2026-10-03T00:00:00+00:00"}])
    sd = d / "steps" / "00-kl"
    sd.mkdir(parents=True)
    out = sd / "stdout"
    out.write_text("[kl-ladder] r1 x p\n    (resumed from saved record)\n    KL 1.0 +/- 0.1\n"
                   "[kl-ladder] r1 x c\n    KL 2.0 +/- 0.1\n[kl-ladder] r2 x p\n")
    born = getattr(os.stat(out), "st_birthtime", None)
    mtime = out.stat().st_mtime
    if born is None:
        pytest.skip("no st_birthtime here")
    os.utime(out, (mtime + 600, mtime + 600))           # one fresh cell took ~600 s
    e = queue_eta.step_eta(st, {"status": "running"}, sd, tmp_path, d, now=mtime + 700)
    per = (mtime + 600 - born) / 1
    assert e["eta_s"] == round(max(0, per * 2 - 100))
    assert "2/4 cells" in e["basis"]


def test_eta_from_history_and_unknown(tmp_path):
    st = {"name": "pack", "cmd": "pack", "args": ["--x", "1"]}
    for i, sec in enumerate((100, 300, 200)):
        _queue(tmp_path, f"old{i}", [st], [{"name": "pack", "status": "pass", "seconds": sec}],
               status="passed")
    _queue(tmp_path, "oldpf", [st], [{"name": "pack", "status": "pass", "seconds": 5}],
           preflight=True, status="passed")
    d = _queue(tmp_path, "now", [st, {"name": "other", "cmd": "smoke", "args": []}],
               [{"name": "pack", "status": "pending"}, {"name": "other", "status": "pending"}])
    e = queue_eta.queue_eta(d, tmp_path, now=time.time())
    assert e["steps"][0]["eta_s"] == 200 and "median of 3" in e["steps"][0]["basis"]
    assert e["steps"][1]["eta_s"] is None
    assert e["eta_s"] is None, "one unknown step makes the queue's ETA unknown, not a guess"
    assert queue_eta.fmt(None) == "ETA unknown"


def test_status_prints_eta(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VQLAB_QUEUE_DIR", str(tmp_path))
    st = {"name": "s", "cmd": "smoke", "args": []}
    d = _queue(tmp_path, "20261003-a", [st], [{"name": "s", "status": "pending"}])
    rq.status(d)
    assert "ETA unknown" in capsys.readouterr().out
    q = ms.t_queue_status()
    assert q["queues"][0]["eta"] == "ETA unknown" and q["queues"][0]["eta_s"] is None
