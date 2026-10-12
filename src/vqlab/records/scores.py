"""The scores index: every KL cell the lab has measured, one line each.

    <records>/registry/scores.jsonl

`vqlab kl-ladder` appends a line per freshly scored rung x corpus (never for
--preflight, which is not a result). A line names its ARTIFACT by the
registry's own identity (provenance.shard_fingerprint: the weight bytes;
runtime: the model.py it ran) and its INSTRUMENT by the teacher cache
(corpus, tokens, chunk, and the cache folder), so a comparison row can say
what produced it (AGENTS.md rule III). The GUI's Library reads it through
`latest()`: the newest line per (artifact fingerprint, runtime, cache).
"""
from __future__ import annotations

import json
import pathlib
import time

from vqlab import config
from vqlab.util.fs import read_jsonl


def path() -> pathlib.Path:
    return config.registry_dir() / "scores.jsonl"


def key(line: dict) -> tuple:
    """What makes two lines the same measurement: bytes, runtime, instrument
    and the number itself."""
    return (line.get("fingerprint"), line.get("runtime"), line.get("cache"), line.get("kl_mnats"))


def append(measured: dict, corpus: str, cache: dict, rec: dict, seen: set | None = None) -> bool:
    """One scored cell. `measured` is provenance.measured(rung dir); `corpus`
    the cache's name (prose / code / lit); `cache` the kl-ladder cache meta;
    `rec` the scorer's record. With `seen` (keys of lines already indexed),
    a cell already there is not written twice; returns whether it was."""
    line = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "artifact": config.portable(measured.get("pin", {}).get("source") or measured["path"]),
        "fingerprint": measured.get("fingerprint"),
        "runtime": (measured.get("runtime") or {}).get("model_py_md5"),
        "build_record": measured.get("build_record"),
        "run_id": measured.get("run_id"),
        "corpus": corpus,
        "cache": config.portable(cache["dir"]),
        "tokens": cache.get("tokens"), "chunk": cache.get("chunk"),
        "kl_mnats": rec.get("mean_kl_millinats"), "kl_sem_mnats": rec.get("kl_sem_millinats"),
        "ppl": rec.get("ppl"), "top1": rec.get("top1_agreement"),
    }
    if seen is not None:
        if key(line) in seen:
            return False
        seen.add(key(line))
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(line) + "\n")
    return True


def read() -> list[dict]:
    """Every line of the index. An unreadable index reads as empty and a torn
    or non-object line is skipped: the index is a convenience, the
    campaigns' own outputs are the record."""
    try:
        rows = read_jsonl(path())
    except OSError:
        return []
    return [r for r in rows if isinstance(r, dict)]


def latest() -> list[dict]:
    """The newest line per (fingerprint, runtime, cache): a rescore of the
    same bytes on the same instrument replaces the older number. A line
    whose bytes were never identified names no artifact, so it is skipped."""
    keep = {}
    for r in read():
        fp = r.get("fingerprint")
        if not fp or str(fp).startswith("unreadable"):
            continue
        keep[(r.get("fingerprint"), r.get("runtime"), r.get("cache"))] = r
    return list(keep.values())
