"""vqlab.util: the shared helpers, and the old private names that now point
at them (kept for one release)."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os

from vqlab.util import clock, fs, hashing, log


def test_sha256_file_whole_head_md5_and_missing(tmp_path):
    p = tmp_path / "f.bin"
    data = os.urandom(3 << 20)
    p.write_bytes(data)
    assert hashing.sha256_file(p) == hashlib.sha256(data).hexdigest()
    assert hashing.sha256_file(p, 1 << 20) == hashlib.sha256(data[:1 << 20]).hexdigest()
    assert hashing.sha256_file(p, algo="md5") == hashlib.md5(data).hexdigest()
    assert hashing.sha256_file(tmp_path / "nope", missing_ok=True) is None
    try:
        hashing.sha256_file(tmp_path / "nope")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing file must raise without missing_ok")


def test_git_blob_sha1_matches_git(tmp_path):
    p = tmp_path / "x"
    p.write_bytes(b"hello\n")
    assert hashing.git_blob_sha1(p) == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_fingerprint_shape_and_provenance_alias(tmp_path):
    from vqlab.records import provenance
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "a.safetensors").write_bytes(b"x" * 10)
    (tmp_path / provenance.RECORD).write_text(json.dumps({"id": "abc"}))
    fp = hashing.fingerprint(tmp_path)
    assert fp["provenance_id"] == "abc"
    assert fp["shards"]["a.safetensors"]["bytes"] == 10
    assert fp["files"]["config.json"] == hashlib.sha256(b"{}").hexdigest()
    assert provenance.fingerprint(tmp_path) == fp
    assert provenance._sha is hashing.sha256_file
    assert provenance.HEAD == hashing.HEAD and provenance.RECORD == hashing.RECORD


def test_clock():
    n = clock.now()
    assert n.tzinfo is not None and abs((dt.datetime.now(dt.timezone.utc) - n).total_seconds()) < 5
    s = clock.now_iso()
    assert dt.datetime.fromisoformat(s).tzinfo is not None and "." not in s
    assert clock.slug("a b/c:d.e-f") == "a_b_c_d.e-f"
    assert len(clock.slug("x" * 100)) == 40


def test_log(capsys):
    log.make_log("[t]")("hi", 1)
    log.make_log(stamp=True)("there")
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "[t] hi 1"
    assert out[1].startswith("[") and out[1].endswith("] there")


def test_dir_bytes(tmp_path):
    (tmp_path / "a.safetensors").write_bytes(b"x" * 5)
    (tmp_path / "b.json").write_bytes(b"y" * 3)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.safetensors").write_bytes(b"z" * 7)
    (tmp_path / "link.safetensors").symlink_to(tmp_path / "a.safetensors")
    assert fs.dir_bytes(tmp_path) == 15
    assert fs.dir_bytes(tmp_path, suffixes=(".safetensors",)) == 12
    assert fs.dir_bytes(tmp_path, suffixes=(".safetensors",), recursive=False) == 5
    assert fs.dir_bytes(tmp_path / "missing", recursive=False) == 0


def test_old_names_still_answer(tmp_path):
    from vqlab.agents import mcp_server, queue_eta, reserve, run_queue
    from vqlab.records import runlog, scratch_cmd
    from vqlab.ship import release_prep
    assert run_queue._slug("a b") == queue_eta._slug("a b") == "a_b"
    assert isinstance(reserve._now(), dt.datetime)
    assert isinstance(runlog._now(), str) and isinstance(run_queue._now(), str)
    (tmp_path / "a.safetensors").write_bytes(b"x" * 4)
    assert mcp_server._dir_bytes(tmp_path) == scratch_cmd._dir_bytes(tmp_path) == 4
    assert release_prep._git_blob(tmp_path / "a.safetensors") == hashing.git_blob_sha1(
        tmp_path / "a.safetensors")
