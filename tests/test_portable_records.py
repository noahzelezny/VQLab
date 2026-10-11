"""Build records and published files must not carry this machine's paths.

Every artifact's vqlab_provenance.json / history (and the 397B rungs'
config.json vq_skipzero.source) recorded absolute local paths, and publish
uploaded them. These tests use temporary roots only.
"""
import hashlib
import json

import pytest

from vqlab import config
from vqlab.records import provenance
from vqlab.ship import leak_guard, publish

# an RFC 1918 address, assembled so no literal address sits in the source
LAN = ".".join(["192", "168", "0", "9"])


@pytest.fixture
def roots(tmp_path, monkeypatch):
    home = tmp_path / "home"
    r = {k: tmp_path / "vol" / k for k in ("models", "scratch", "teachers", "fits")}
    for p in [home, *r.values()]:
        p.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("VQLAB_CONFIG", str(tmp_path / "none.toml"))
    monkeypatch.setenv("VQLAB_MODELS_DIR", str(r["models"]))
    monkeypatch.setenv("VQLAB_SCRATCH", str(r["scratch"]))
    monkeypatch.setenv("VQLAB_TEACHERS_DIR", str(r["teachers"]))
    monkeypatch.setenv("VQLAB_FIT_STORE", str(r["fits"]))
    monkeypatch.delenv("VQLAB_ROOTS", raising=False)
    r["home"] = home
    return r


def test_portable_text(roots):
    m, s = roots["models"], roots["scratch"]
    assert config.portable_text(f"{m}/A") == "<models>/A"
    assert config.portable_text(f"--out={s}/x/y") == "--out=<scratch>/x/y"
    assert config.portable_text(f"{roots['home']}/q/tree") == "~/q/tree"
    assert config.portable_text(f"{m}X/A") == f"{m}X/A"        # not a prefix match
    assert config.portable_text("no paths here") == "no paths here"
    assert config.resolve_recorded("<models>/A") == f"{m}/A"
    assert config.resolve_recorded(f"{m}/A") == f"{m}/A"         # old records


def _art(roots, name="A"):
    a = roots["models"] / name
    a.mkdir()
    (a / "config.json").write_text('{"x": 1}')
    return a


def test_new_record_is_portable(roots):
    teacher = roots["teachers"] / "T"
    teacher.mkdir()
    (teacher / "config.json").write_text("{}")
    a = _art(roots)
    rec = provenance.write_build_record(a, tool="t", argv=["t", str(a)],
                                        inputs=[("teacher", teacher)])
    text = (a / provenance.RECORD).read_text()
    assert str(roots["home"]) not in text and str(roots["models"]) not in text
    assert "repo" not in rec["code"] and "version" in rec["code"]
    assert rec["inputs"][0]["path"] == "<teachers>/T"
    assert provenance.verify(a, rec) == []


def _old_record(roots, a, parent):
    """A record as written before this change: absolute paths, code.repo."""
    rec = provenance.write_build_record(a, tool="t")
    repo = str(roots["home"] / "src" / "lab")
    rec["tool"]["argv"] = ["t", str(a), f"--src={parent}"]
    rec["tool"]["cwd"] = repo
    rec["env"]["host"] = "builder-box"
    rec["code"]["repo"] = repo
    rec["inputs"] = [{"role": "base", "path": str(parent), "realpath": str(parent),
                      "shards": {"m.safetensors": [1, "ab" * 32]}}]
    rec["id"] = hashlib.sha256(provenance._canon(rec).encode()).hexdigest()
    (a / provenance.RECORD).write_text(json.dumps(rec, indent=1, default=str))
    (a / provenance.HISTORY).write_text(json.dumps(rec, sort_keys=True) + "\n")
    return rec


def test_make_portable_paths_only(roots):
    parent = _art(roots, "P")
    provenance.write_build_record(parent, tool="p")
    a = _art(roots)
    (a / "config.json").write_text(
        '{\n  "b": 2,\n  "vq_skipzero": {\n    "source": "%s",\n    "incremental": '
        '[{"source": "%s", "n": 1.50}]\n  },\n  "a": 1\n}\n' % (parent, parent))
    old = _old_record(roots, a, parent)
    cfg_before = (a / "config.json").read_text()

    dry = provenance.make_portable(a, dry_run=True)
    assert {p.name for p, _, _ in dry} == {provenance.RECORD, provenance.HISTORY, "config.json"}
    assert json.loads((a / provenance.RECORD).read_text()) == old       # dry run wrote nothing

    provenance.make_portable(a)
    new = provenance.load(a)
    assert (a / (provenance.RECORD + ".bak")).exists()
    # config.json changed, so the record is AMENDED: new id, previous -> the
    # (portable) old record, which is in history; only config.json's hash moved
    assert new["previous"] == old["id"] and new["tool"]["note"] == provenance.SANITIZE_NOTE
    assert provenance.verify(a, new) == []
    hist = [json.loads(x) for x in (a / provenance.HISTORY).read_text().splitlines()]
    prior = hist[-1]
    assert prior["id"] == old["id"] and provenance.id_ok(prior) and "portable_id" in prior
    assert "host" not in prior["env"] and "host" not in hist[0]["env"]
    assert prior["code"]["repo"] == config.CHECKOUT_LABEL
    assert prior["tool"]["cwd"] == config.CHECKOUT_LABEL
    assert new["inputs"][0]["shards"] == old["inputs"][0]["shards"]
    assert {k: v for k, v in new["outputs"].items() if k != "config.json"} == \
        {k: v for k, v in old["outputs"].items() if k != "config.json"}
    assert new["outputs"]["config.json"]["sha256"] != old["outputs"]["config.json"]["sha256"]
    hits = leak_guard.scan([a / provenance.RECORD, a / provenance.HISTORY, a / "config.json"], a)
    assert hits == []
    cfg = (a / "config.json").read_text()
    assert cfg == cfg_before.replace(str(parent), "<models>/P")          # every other byte
    # readers: the portable parent path still resolves
    assert config.resolve_recorded(new["inputs"][0]["realpath"]) == str(parent)
    assert provenance.make_portable(a) == []                             # idempotent


def test_tampered_portable_record_fails(roots):
    a = _art(roots)
    old = _old_record(roots, a, roots["models"])
    provenance.make_portable(a)
    rec = provenance.load(a)
    assert rec["id"] == old["id"] and provenance.id_ok(rec)   # no config change: id kept
    assert "host" not in rec["env"]
    rec["method"] = {"edited": True}
    assert not provenance.id_ok(rec)


def test_leak_guard(roots, tmp_path):
    d = tmp_path / "pub"
    d.mkdir()
    (d / "ok.md").write_text("version 10.0 and <models>/A\n")
    (d / "home.json").write_text(f'{{"p": "{roots["home"]}/x"}}\n')
    (d / "root.txt").write_text(f"a\n{roots['scratch']}/run\n")
    (d / "lan.py").write_text(f"HOST = '{LAN}'\n")
    (d / "bin.safetensors").write_bytes(str(roots["home"]).encode())
    hits = {(f, n) for f, n, _ in leak_guard.scan(sorted(d.iterdir()), d)}
    assert hits == {("home.json", 1), ("root.txt", 2), ("lan.py", 1)}


def test_leak_guard_hosts(roots, tmp_path, monkeypatch):
    monkeypatch.setattr(leak_guard, "host_names", lambda: {"box-a"})
    d = tmp_path / "pub"
    d.mkdir()
    (d / "a.md").write_text("ran on box-a today\nbox-ab is fine\n")
    (d / "vqlab_provenance.json").write_text('{\n "host": "renamed-box"\n}\n')
    hits = {(f, n) for f, n, _ in leak_guard.scan(sorted(d.iterdir()), d)}
    assert hits == {("a.md", 1), ("vqlab_provenance.json", 2)}


def test_new_record_has_no_host():
    assert "host" not in provenance.env_state()


def test_backups_excluded():
    for n in ("model.py.pre-rt-2026-10-09", "tokenizer.json.pre-tok-2026-10-10",
              "config.json.bak", "x.tmp"):
        assert leak_guard.is_backup(n)
    for n in ("model.py", "config.json", "tokenizer.json"):
        assert not leak_guard.is_backup(n)


def test_publish_refuses_leaks_before_gate(roots, monkeypatch, capsys):
    a = _art(roots)
    (a / "README.md").write_text(f"built in {roots['scratch']}/F1\n")
    (a / "model.py.pre-x").write_text("")
    ran = []
    monkeypatch.setattr(publish.subprocess, "run", lambda *x, **k: ran.append(x))
    assert publish.main(["--artifact", str(a), "--repo", "o/n", "--files", "README.md"]) == 1
    out = capsys.readouterr().out
    assert "README.md:1" in out and "provenance --portable" in out and not ran
    assert publish.main(["--artifact", str(a), "--repo", "o/n", "--files", "model.py.pre-x"]) == 1
    assert "backup" in capsys.readouterr().out
