#!/usr/bin/env python3
"""Write bench/jobs.json (the /api/queues shape + snapshot time + gpu) and
refresh the `const JOBS_SNAPSHOT=...;` line in bench.html, then regenerate
index.html. Public repo: `source` is reduced to its basename and the
host name is dropped.

    python src/vqlab/agents/gui_static/bench/make_jobs_snapshot.py [--n 40]
"""
import argparse
import datetime
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))  # src/
from vqlab.agents import gui  # noqa: E402

HEAD = ('<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    n = ap.parse_args().n
    runs = gui.queues_snapshot(n)
    scrub = lambda t: re.sub(r"/(?:Users|Volumes|private|home)/[^:)]*/([^/:)\s]+)", r"\1", t)  # noqa: E731
    for r in runs:
        r.pop("host", None)
        if r.get("source"):
            r["source"] = pathlib.PurePath(r["source"]).name
        for st in r["steps"]:
            for k in ("reasons", "warnings", "tail"):
                st[k] = [scrub(x) for x in st.get(k) or []]
            if st.get("cmd"):
                st["cmd"] = scrub(st["cmd"])
    gpu = None
    try:
        import importlib
        g = importlib.import_module("vqlab.agents.mcp_server").t_gpu_state()
        h = g.get("lease_holder")
        gpu = {"held": bool(h), "by": (h.get("holder") or h.get("cmd") or "unknown") if h else None}
    except Exception:  # noqa: BLE001
        pass
    snap = {"snapshot": datetime.datetime.now().isoformat(timespec="seconds"), "gpu": gpu, "queues": runs}
    (HERE / "jobs.json").write_text(json.dumps(snap, indent=1) + "\n")
    html = (HERE / "bench.html").read_text()
    line = "const JOBS_SNAPSHOT=" + json.dumps(snap, separators=(",", ":")) + ";"
    html, k = re.subn(r"^const JOBS_SNAPSHOT=.*;$", lambda m: line, html, flags=re.M)
    if not k:
        sys.exit("bench.html has no `const JOBS_SNAPSHOT=...;` line")
    (HERE / "bench.html").write_text(html)
    (HERE / "index.html").write_text(HEAD + html)
    print(f"{len(runs)} queues -> jobs.json, bench.html, index.html")


if __name__ == "__main__":
    main()
