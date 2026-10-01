#!/usr/bin/env python3
"""vqlab gui — a local, READ-ONLY window onto the lab.

    vqlab gui [--port 8781] [--open]

Serves one page on 127.0.0.1 (never another interface) over data the lab
already keeps -- nothing here computes a number of its own:

  Fleet      registry/artifacts.jsonl + registry/hub.jsonl (runtime profile,
             geometry mix, size, Hub drift, build record), lineage on click
  Fit store  <store>/index.jsonl: per teacher, a layer x projection grid of
             which (d, K) are already fitted -- what a mixed-codebook build
             can be assembled from without paying for a k-means
  Families   families/*/teachers/*/{profile,onboard}.json
  Runs       ~/.vqlab/runs.jsonl, the MCP run dirs, and who holds the GPU
  Queues     /api/queues: recent `vqlab queue` runs (state.json per run, with
             `live` = the runner pid is alive, the check `queue status` uses)
  Bench      /bench/ serves gui_static/bench/ (the Library/Recipe/Jobs page)
             on the same origin, so its Jobs page reads /api/queues live

Stdlib only; reads index files and small JSON, never a tensor, so it is safe
on a box that is mid-experiment. Every endpoint is a GET; there is no way to
launch, delete or write anything from the page. Launching stays with the CLI
and MCP `run`, where the GPU lease and exo gates live.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import pathlib
import sys
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # src/
from vqlab import _layout  # noqa: E402,F401  one module object per name

REPO = _layout.SRC.parent
STATIC = pathlib.Path(__file__).resolve().parent / "gui_static"


def _jsonl(p, limit=None):
    p = pathlib.Path(p)
    if not p.exists():
        return []
    lines = p.read_text().splitlines()
    if limit:
        lines = lines[-limit:]
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def _families_dir():
    return pathlib.Path(os.environ.get("VQLAB_FAMILIES_DIR") or REPO / "families")


# ---------------------------------------------------------------- endpoints
def api_fleet(_q):
    arts = _jsonl(REPO / "registry" / "artifacts.jsonl")
    hub = {h["repo"]: h for h in _jsonl(REPO / "registry" / "hub.jsonl")}
    twins = {}
    for e in arts:
        twins.setdefault(e["fingerprint"], []).append(e["name"])
    for e in arts:
        e["hub"] = hub.get(e.get("hf_repo") or "")
        e["same_bytes_as"] = [n for n in twins[e["fingerprint"]] if n != e["name"]]
    return {"artifacts": arts}


def api_provenance(q):
    """Build record + lineage for a REGISTERED artifact only: the page can
    never be used to read an arbitrary path."""
    path = q.get("path", [""])[0]
    known = {e["path"] for e in _jsonl(REPO / "registry" / "artifacts.jsonl")}
    if path not in known:
        return {"error": "not a registered artifact"}
    import provenance
    try:
        rec = provenance.load(path)
    except FileNotFoundError:
        return {"record": None}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        provenance.lineage(path)
    return {"record": rec, "summary": provenance.summary(path, rec),
            "lineage": buf.getvalue(), "verify": [list(x) for x in provenance.verify(path, rec)]
            if q.get("verify") else None}


def api_fits(q):
    import fitstore
    roots = fitstore.roots()
    recs = {}
    for r in roots:
        recs.update(fitstore.read_index(r))
    fam, teacher = q.get("family", [None])[0], q.get("teacher", [None])[0]
    if not fam:                                  # summary: which teachers exist
        tree = {}
        for r in recs.values():
            t = tree.setdefault(r["family"], {}).setdefault(r["teacher"], {"fits": 0, "geoms": {}})
            t["fits"] += 1
            g = f"d{r['d']}-K{r['K']}"
            t["geoms"][g] = t["geoms"].get(g, 0) + 1
        return {"roots": [str(r) for r in roots], "total": len(recs), "tree": tree}
    rows = [r for r in recs.values() if r["family"] == fam and r["teacher"] == teacher]
    slim = [{"fit_id": r["fit_id"], "layer": r.get("layer"), "proj": r["proj"],
             "d": r["d"], "K": r["K"], "bytes": r["bytes"], "module": r["module"],
             "run_dir": r.get("source", {}).get("run_dir"),
             "recipe": r.get("recipe"), "location": r.get("location")} for r in rows]
    prof = _families_dir() / fam / "teachers" / teacher / "profile.json"
    layers = json.loads(prof.read_text())["arch"]["vq_layers"] if prof.exists() else None
    return {"fits": slim, "layers": layers}


def api_families(_q):
    out = []
    root = _families_dir()
    for fam in sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []:
        entry = fam / "entry.json"
        teachers = []
        for t in sorted((fam / "teachers").glob("*")) if (fam / "teachers").is_dir() else []:
            prof = t / "profile.json"
            onb = t / "onboard.json"
            teachers.append({
                "teacher": t.name,
                "profile": json.loads(prof.read_text()) if prof.exists() else None,
                "onboard": json.loads(onb.read_text()) if onb.exists() else None,
                "research": sorted(f.name for f in (t / "research").glob("*"))
                if (t / "research").is_dir() else []})
        out.append({"family": fam.name, "data_entry": entry.exists(), "teachers": teachers,
                    "docs": sorted(f.name for f in fam.glob("*.md"))})
    return {"families": out}


def api_runs(q):
    n = int(q.get("n", ["200"])[0])
    log = pathlib.Path(os.environ.get("VQLAB_LOG_DIR", pathlib.Path.home() / ".vqlab")) / "runs.jsonl"
    starts, ends = {}, {}
    for r in _jsonl(log, limit=n * 3):
        (starts if r.get("event") == "start" else ends)[r.get("run_id")] = r
    runs = []
    for rid, s in list(starts.items())[-n:]:
        e = ends.get(rid)
        runs.append({"run_id": rid, "parent": s.get("parent_run_id"), "time": s["time"],
                     "cmd": s["cmd"], "argv": s.get("argv", []),
                     "commit": (s.get("code") or {}).get("commit"),
                     "dirty": (s.get("code") or {}).get("dirty"),
                     "rc": e.get("rc") if e else None,
                     "seconds": e.get("seconds") if e else None, "ended": bool(e)})
    mcp, gpu = [], None
    try:
        import importlib
        ms = importlib.import_module("vqlab.agents.mcp_server")
        gpu = ms.t_gpu_state()
        mcp = ms.t_list_runs(40).get("runs", [])
    except Exception as e:                       # noqa: BLE001
        gpu = {"error": f"{type(e).__name__}: {e}"}
    return {"runs": list(reversed(runs)), "mcp": mcp, "gpu": gpu}


def queues_snapshot(n=40):
    """Most recent queue runs, newest first. `live` reuses the runner-alive
    check `vqlab queue status` makes (mcp_server._pid_alive on state.pid)."""
    import importlib
    ms = importlib.import_module("vqlab.agents.mcp_server")
    rq = importlib.import_module("vqlab.agents.run_queue")
    dirs = sorted((p for p in rq.queues_dir().glob("*") if (p / "state.json").exists()),
                  key=lambda p: p.name, reverse=True)[:max(1, n)]
    out = []
    for d in dirs:
        try:
            s = json.loads((d / "state.json").read_text())
        except ValueError:
            continue
        out.append({"id": d.name, "name": s.get("name"), "status": s.get("status"),
                    "live": bool(ms._pid_alive(s.get("pid"))), "preflight": bool(s.get("preflight")),
                    "created": s.get("created"), "host": s.get("host"), "commit": s.get("commit"),
                    "source": s.get("source"),
                    "steps": [{k: r.get(k) for k in ("name", "status", "seconds", "started", "finished",
                                                     "rc", "attempts", "reasons", "warnings")}
                              for r in s.get("steps", [])]})
    return out


def api_queues(q):
    return queues_snapshot(int(q.get("n", ["40"])[0]))


ROUTES = {"/api/fleet": api_fleet, "/api/provenance": api_provenance,
          "/api/fits": api_fits, "/api/families": api_families, "/api/runs": api_runs,
          "/api/queues": api_queues}


BENCH_TYPES = {".html": "text/html; charset=utf-8", ".json": "application/json",
               ".css": "text/css", ".js": "text/javascript", ".svg": "image/svg+xml",
               ".png": "image/png"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_a):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _bench(self, rel):
        root = (STATIC / "bench").resolve()
        f = (root / urllib.parse.unquote(rel)).resolve()
        ctype = BENCH_TYPES.get(f.suffix)
        if root not in f.parents or ctype is None or not f.is_file():
            return self._send(404, b'{"error":"not found"}', "application/json")
        self._send(200, f.read_bytes(), ctype)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path in ("/", "/index.html"):
            return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        if u.path == "/bench":
            self.send_response(301)
            self.send_header("Location", "/bench/")
            self.end_headers()
            return
        if u.path.startswith("/bench/"):
            return self._bench(u.path[len("/bench/"):] or "index.html")
        fn = ROUTES.get(u.path)
        if fn is None:
            return self._send(404, b'{"error":"not found"}', "application/json")
        try:
            data = fn(urllib.parse.parse_qs(u.query))
            code = 200
        except Exception as e:                   # noqa: BLE001
            data, code = {"error": f"{type(e).__name__}: {e}"}, 500
        self._send(code, json.dumps(data, default=str).encode(), "application/json")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vqlab gui", description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=8781)
    ap.add_argument("--open", action="store_true", help="open it in the default browser")
    a = ap.parse_args(argv)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://127.0.0.1:{a.port}/"
    print(f"vqlab gui (read-only) on {url}  (ctrl-c to stop)", flush=True)
    if a.open:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
