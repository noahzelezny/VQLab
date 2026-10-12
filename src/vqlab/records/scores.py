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


def path() -> pathlib.Path:
    return config.registry_dir() / "scores.jsonl"


def append(measured: dict, corpus: str, cache: dict, rec: dict) -> None:
    """One scored cell. `measured` is provenance.measured(rung dir); `corpus`
    the cache's name (prose / code / lit); `cache` the kl-ladder cache meta;
    `rec` the scorer's record."""
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
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(line) + "\n")


def read() -> list[dict]:
    try:
        lines = path().read_text().splitlines()
    except OSError:
        return []
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except ValueError:
            continue
    return out


def latest() -> list[dict]:
    """The newest line per (fingerprint, runtime, cache): a rescore of the
    same bytes on the same instrument replaces the older number."""
    keep = {}
    for r in read():
        keep[(r.get("fingerprint"), r.get("runtime"), r.get("cache"))] = r
    return list(keep.values())
