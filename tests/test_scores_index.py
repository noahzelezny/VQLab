"""records/scores.py: kl-ladder's measured cells, one line each, naming the
artifact and the instrument; the newest per (bytes, runtime, cache) wins."""
from vqlab.records import scores


def _m(path, fp="f1", md5="r1"):
    return {"path": path, "fingerprint": fp, "runtime": {"model_py_md5": md5}, "build_record": "b"}


def test_append_and_latest(tmp_path, monkeypatch):
    monkeypatch.setenv("VQLAB_RECORDS", str(tmp_path))
    cache = {"dir": str(tmp_path / "c-prose"), "tokens": 12287, "chunk": 512}
    scores.append(_m("/a"), "prose", cache, {"mean_kl_millinats": 50.0, "kl_sem_millinats": 1.0,
                                              "ppl": 2.0, "top1_agreement": 0.9})
    scores.append(_m("/a"), "prose", cache, {"mean_kl_millinats": 49.0})          # a rescore
    scores.append(_m("/a", md5="r2"), "prose", cache, {"mean_kl_millinats": 48.0})  # new runtime
    (tmp_path / "registry" / "scores.jsonl").open("a").write("not json\n")
    rows = sorted(scores.latest(), key=lambda r: r["runtime"])
    assert [(r["runtime"], r["kl_mnats"]) for r in rows] == [("r1", 49.0), ("r2", 48.0)]
    r = rows[0]
    assert r["corpus"] == "prose" and r["fingerprint"] == "f1" and r["chunk"] == 512


def test_a_pin_is_recorded_as_its_source(tmp_path, monkeypatch):
    monkeypatch.setenv("VQLAB_RECORDS", str(tmp_path))
    m = dict(_m(str(tmp_path / "pin")), pin={"source": str(tmp_path / "art")})
    scores.append(m, "code", {"dir": str(tmp_path / "c")}, {"mean_kl_millinats": 1.0})
    assert scores.read()[0]["artifact"] == str(tmp_path / "art")


def test_no_index_is_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("VQLAB_RECORDS", str(tmp_path))
    assert scores.latest() == []
