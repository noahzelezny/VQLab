"""`vqlab scratch reclaimable`: lists an unpacked dir only when its packed
counterpart's build record names it AND both sides still verify; prints the
rm lines; never deletes. Each refusal is gated in both directions on the same
fixture (a gate must fail on a known-bad input before its pass means
anything)."""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

from vqlab.records import provenance
from vqlab.agents import scratch as scratch_cmd

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"


def _art(d: pathlib.Path, shard_bytes: bytes):
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    (d / "model-00001-of-00001.safetensors").write_bytes(shard_bytes)
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))


def _fixture(root: pathlib.Path):
    fit, packed, other = root / "fit-k2048", root / "fit-k2048-packed", root / "fit-k256"
    _art(fit, b"u" * 4000)
    provenance.write_build_record(fit, tool="fit-moe")
    _art(packed, b"p" * 2750)
    provenance.write_build_record(packed, tool="pack", inputs=[("src", fit)])
    _art(other, b"x" * 100)                     # unpacked, never packed: not listed
    provenance.write_build_record(other, tool="fit-moe")
    return fit, packed, other


def test_lists_verified_pairs_and_prints_rm(tmp_path, capsys):
    fit, packed, other = _fixture(tmp_path)
    cands, kept = scratch_cmd.reclaimable([tmp_path])
    assert [c["path"] for c in cands] == [str(fit)] and not kept
    assert cands[0]["packed"] == str(packed)
    assert cands[0]["bytes"] == provenance_bytes(fit)
    capsys.readouterr()                                      # drop the fixture's PROVENANCE lines
    scratch_cmd.main(["reclaimable", "--root", str(tmp_path)])
    out = capsys.readouterr().out
    assert f"rm -rf {fit}" in out and "Nothing was deleted" in out
    assert f"rm -rf {other}" not in out and f"KEPT [unclassified] {other}" in out
    assert fit.is_dir() and other.is_dir()                   # nothing deleted
    cands_full, _ = scratch_cmd.reclaimable([tmp_path], full=True)
    assert [c["path"] for c in cands_full] == [str(fit)]


def provenance_bytes(d):
    return sum(f.stat().st_size for f in d.iterdir() if f.is_file() and not f.is_symlink())


def test_refuses_when_packed_does_not_verify(tmp_path):
    fit, packed, _ = _fixture(tmp_path)
    (packed / "model-00001-of-00001.safetensors").write_bytes(b"p" * 10)   # truncated pack
    cands, kept = scratch_cmd.reclaimable([tmp_path])
    assert not cands and "does not verify" in kept[0]["reason"]


def test_refuses_when_source_changed(tmp_path):
    fit, packed, _ = _fixture(tmp_path)
    (fit / "model-00001-of-00001.safetensors").write_bytes(b"v" * 4000)    # same size, new bytes
    cands, kept = scratch_cmd.reclaimable([tmp_path])
    assert not cands and "changed" in kept[0]["reason"]


def test_refuses_when_something_links_into_source(tmp_path):
    fit, packed, _ = _fixture(tmp_path)
    pin = tmp_path / "pin"
    pin.mkdir()
    os.symlink(fit / "model-00001-of-00001.safetensors", pin / "model-00001-of-00001.safetensors")
    cands, kept = scratch_cmd.reclaimable([tmp_path])
    assert not cands and "symlink" in kept[0]["reason"]


def test_cli_routes_and_has_no_delete(tmp_path):
    _fixture(tmp_path)
    env = {**os.environ, "PYTHONPATH": str(SRC), "VQLAB_SCRATCH": str(tmp_path),
           "VQLAB_CONFIG": str(tmp_path / "none.toml"), "VQLAB_LOG_DIR": str(tmp_path / "log")}
    p = subprocess.run([sys.executable, "-m", "vqlab.cli", "scratch", "reclaimable"],
                       capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stderr
    assert "rm -rf" in p.stdout and (tmp_path / "fit-k2048").is_dir()
    p = subprocess.run([sys.executable, "-m", "vqlab.cli", "scratch", "reclaimable", "--delete"],
                       capture_output=True, text=True, env=env)
    assert p.returncode != 0


# ---------------------------------------------------------- the wider classes
def _st(path: pathlib.Path, tensors: dict[str, tuple[str, list, bytes]]):
    """A minimal safetensors file: {name: (dtype, shape, raw bytes)}."""
    import struct
    hdr, blobs, off = {}, b"", 0
    for k, (dt, shape, raw) in tensors.items():
        hdr[k] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + len(raw)]}
        blobs += raw
        off += len(raw)
    hj = json.dumps(hdr).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(hj)) + hj + blobs)


def _fit(path: pathlib.Path, layer: int, seed: bytes):
    m = f"model.layers.{layer}.mlp.switch_mlp.up_proj"
    _st(path, {m + ".codebook": ("F16", [4, 2], seed * 16),
               m + ".codes": ("U8", [1, 2, 64], b"\x01" * 128),
               m + ".vq_scales": ("F16", [1, 2, 1], b"\x00" * 4)})
    return m


def _filed_index(tmp_path, *parts):
    from vqlab.core.source import fit_store
    store = tmp_path / "store"
    for f, m in parts:
        fit_store.put(f, m, "fam", "teacher", root=store)
    return fit_store.read_index(store)


def test_filed_fit_dir_listed_and_unfiled_kept(tmp_path):
    root = tmp_path / "scratch"
    a = root / "run_parts" / "a.safetensors"
    ma = _fit(a, 1, b"a")
    index = _filed_index(tmp_path, (a, ma))
    for full in (False, True):
        items, kept = scratch_cmd.fit_items([root], [], full=full, index=index)
        assert [i.path for i in items] == [str(root / "run_parts")] and not kept
        assert "1/1 modules" in items[0].evidence
    _fit(root / "run_parts" / "b.safetensors", 2, b"b")           # one module never filed
    items, kept = scratch_cmd.fit_items([root], [], index=index)
    assert not items and "not filed" in kept[0].reason


def test_fit_dir_with_same_codebook_other_codes_kept(tmp_path):
    root = tmp_path / "scratch"
    a = root / "refit" / "a.safetensors"
    m = _fit(a, 1, b"a")
    index = _filed_index(tmp_path, (a, m))
    _st(a, {m + ".codebook": ("F16", [4, 2], b"a" * 16),      # same codebook = same fit_id
            m + ".codes": ("U8", [1, 2, 64], b"\x02" * 128),
            m + ".vq_scales": ("F16", [1, 2, 1], b"\x00" * 4)})
    items, kept = scratch_cmd.fit_items([root], [], index=index)
    assert not items and "differs from the filed copy" in kept[0].reason


def test_fit_dir_with_other_tensors_kept(tmp_path):
    root = tmp_path / "scratch"
    a = root / "mixed" / "a.safetensors"
    m = _fit(a, 1, b"a")
    index = _filed_index(tmp_path, (a, m))
    _st(root / "artifact" / "model.safetensors", {"model.embed_tokens.weight":
                                                  ("F16", [2], b"\x00" * 4)})
    items, kept = scratch_cmd.fit_items([root], [], index=index)
    assert [i.path for i in items] == [str(root / "mixed")] and not kept  # artifact: no fit dir
    _fit(root / "mixed" / "b.safetensors", 1, b"a")                  # a second, filed part ...
    _st(root / "mixed" / "c.safetensors", {"model.embed_tokens.weight":  # ... and a stray tensor
                                           ("F16", [2], b"\x00" * 4)})
    items, kept = scratch_cmd.fit_items([root], [], index=index)
    assert not items and "non-fit tensor" in kept[0].reason


def _cache(d: pathlib.Path, payload: bytes):
    d.mkdir(parents=True)
    (d / scratch_cmd.CACHE_MARK).write_bytes(payload)
    (d / "meta.json").write_text('{"corpus": "prose"}')


def test_duplicate_cache_listed_registered_kept(tmp_path):
    root, reg = tmp_path / "scratch", tmp_path / "scratch" / "registered"
    _cache(reg, b"topk")
    _cache(root / "copy", b"topk")
    _cache(root / "other", b"diff")
    caches = [{"teacher": "T", "corpus": "prose", "path": str(reg)}]
    items, kept = scratch_cmd.cache_items([root], [], caches=caches)
    assert [i.path for i in items] == [str(root / "copy")]
    why = {pathlib.Path(k.path).name: k.reason for k in kept}
    assert "registered cache" in why["registered"]          # duplicated, still never listed
    assert "no registered cache" in why["other"]


def test_preflight_of_finished_queue_listed_running_kept(tmp_path, monkeypatch):
    root, queues = tmp_path / "scratch", tmp_path / "queues"
    pf = root / "queue-preflight"
    monkeypatch.setenv("VQLAB_PREFLIGHT_DIR", str(pf))
    monkeypatch.setenv("VQLAB_QUEUE_DIR", str(queues))
    for name, state in (("q-failed-preflight", {"status": "failed"}),
                        ("q-running-preflight", {"status": "running", "host": "another-box"}),
                        ("q-orphan-preflight", None)):
        (pf / name / "fit").mkdir(parents=True)
        (pf / name / "fit" / "x.bin").write_bytes(b"z" * 10)
        if state:
            (queues / name).mkdir(parents=True)
            (queues / name / "state.json").write_text(json.dumps(state))
    items, kept = scratch_cmd.preflight_items([root], [])
    assert sorted(pathlib.Path(i.path).name for i in items) == ["q-failed-preflight",
                                                               "q-orphan-preflight"]
    assert [pathlib.Path(k.path).name for k in kept] == ["q-running-preflight"]
    assert "running" in kept[0].reason


def test_campaigns_are_archive_not_rm(tmp_path, monkeypatch, capsys):
    root = tmp_path / "scratch"
    (root / "F203-family-m3").mkdir(parents=True)
    (root / "F999-owed").mkdir()
    log = tmp_path / "FINDINGS-LOG.md"
    log.write_text("# log\n\n## F203 (2026-10-07) — rolled flags\n")
    monkeypatch.setenv("VQLAB_FINDINGS_LOG", str(log))
    items, kept = scratch_cmd.campaign_items([root])
    assert [pathlib.Path(i.path).name for i in items] == ["F203-family-m3"]
    assert "line 3" in items[0].evidence
    assert "no entry" in kept[0].reason
    scratch_cmd.main(["reclaimable", "--root", str(root), "--json"])
    rep = json.loads(capsys.readouterr().out)
    assert {i["cls"] for i in rep["items"]} == {"campaigns"} and rep["total_bytes"] == 0
    scratch_cmd.main(["reclaimable", "--root", str(root)])
    out = capsys.readouterr().out
    assert "rm -rf" not in out and "ARCHIVE candidates" in out
    assert "KEPT [campaigns]" in out
