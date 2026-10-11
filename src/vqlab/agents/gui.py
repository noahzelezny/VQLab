#!/usr/bin/env python3
"""vqlab gui — the VQ Lab GUI: Library · Recipe · Jobs.

    vqlab gui [--port 8781] [--open]

Serves gui_static/bench/index.html at / on 127.0.0.1 (never another
interface), plus the JSON it reads, over data the lab already keeps --
nothing here computes a number of its own:

  /api/fleet        <records>/registry/{artifacts,hub}.jsonl
  /api/provenance   build record + lineage of a registered artifact
  /api/fits         <store>/index.jsonl: which (d, K) are fitted per layer
  /api/families     families/ + <families>: profile and onboard state
  /api/runs         ~/.vqlab/runs.jsonl, the MCP run dirs, the GPU holder
  /api/queues       recent `vqlab queue` runs
  /api/settings     storage locations; POST sets one (agents/settings.py)
  /api/scratch      scratch folders; /reclaimable adds rm lines, never run
  /api/storage/usage  per volume: total/used/free, what each vqlab location holds

Stdlib only; reads index files and small JSON, never a tensor, so it is safe
on a box that is mid-experiment. Every endpoint is a GET except one POST
that writes one key of the user's config file (same-origin JSON only; the
server listens on 127.0.0.1). There is no way to launch or delete anything
from the page. Launching stays with the CLI and MCP `run`, where the GPU
lease and exo gates live.
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


from vqlab import config
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
    return config.families_dir()


# ---------------------------------------------------------------- endpoints
def api_fleet(_q):
    arts = _jsonl(config.registry_dir() / "artifacts.jsonl")
    hub = {h["repo"]: h for h in _jsonl(config.registry_dir() / "hub.jsonl")}
    twins = {}
    for e in arts:
        twins.setdefault(e["fingerprint"], []).append(e["name"])
    for e in arts:
        e["hub"] = hub.get(e.get("hf_repo") or "")
        e["same_bytes_as"] = [n for n in twins[e["fingerprint"]] if n != e["name"]]
        e["bundle"] = _bundle_verdict(e["path"])
    return {"artifacts": arts}


def _bundle_verdict(path):
    """check-bundle's verdict on the artifact as it is NOW: PASS (current
    runtime), STALE (an earlier certified runtime: slower, same numbers),
    FAIL, or MISSING. Reads model.py text only."""
    if not pathlib.Path(path, "model.py").exists():
        return {"verdict": "MISSING", "note": "no model.py"}
    from vqlab.gate import check_bundle
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            rc, lines = check_bundle.check(path, strict=True)
    except Exception:
        return {"verdict": "FAIL", "note": "check-bundle crashed"}
    verdict = {0: "PASS", 1: "FAIL", 2: "STALE"}.get(rc, "FAIL")
    note = ""
    if verdict == "STALE":
        # A certified revision whose gains apply only to features this
        # artifact does not use is current in effect: a rebundle changes nothing.
        text = "\n".join(map(str, lines))
        reg = json.loads((pathlib.Path(check_bundle.__file__).parents[1]
                          / "runtime" / "equivalent_revisions.json").read_text())
        cfg = json.loads(pathlib.Path(path, "config.json").read_text())
        for rev in reg["revisions"]:
            if rev["sha256"][:12] in text:
                note = rev.get("gains", "")
                only = rev.get("gains_only_for")
                if only is not None and not any(cfg.get(k) for k in only):
                    verdict = "CURRENT"
                break
    return {"verdict": verdict, "note": note}


def api_provenance(q):
    """Build record + lineage for a REGISTERED artifact only: the page can
    never be used to read an arbitrary path."""
    path = q.get("path", [""])[0]
    known = {e["path"] for e in _jsonl(config.registry_dir() / "artifacts.jsonl")}
    if path not in known:
        return {"error": "not a registered artifact"}
    from vqlab.records import provenance
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
    from vqlab.core.source import fit_store as fitstore
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
    """Families from BOTH places the registry reads (the checkout's
    families/ and the user's families_dir()), merged per family and teacher;
    a teacher's profile or onboard state from either place counts."""
    fams = {}
    for root in (config.checkout_families_dir(), _families_dir()):
        for fam in sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []:
            f = fams.setdefault(fam.name, {"family": fam.name, "data_entry": False,
                                           "teachers": {}, "docs": set()})
            f["data_entry"] |= (fam / "entry.json").exists()
            f["docs"] |= {d.name for d in fam.glob("*.md") if d.name != "CONTEXT.md"}
            for t in sorted((fam / "teachers").glob("*")) if (fam / "teachers").is_dir() else []:
                e = f["teachers"].setdefault(t.name, {"teacher": t.name, "profile": None,
                                                      "onboard": None, "research": set()})
                for key, fn in (("profile", "profile.json"), ("onboard", "onboard.json")):
                    if e[key] is None and (t / fn).exists():
                        e[key] = json.loads((t / fn).read_text())
                if (t / "research").is_dir():
                    e["research"] |= {r.name for r in (t / "research").glob("*")}
    out = []
    for f in fams.values():
        teachers = [dict(t, research=sorted(t["research"])) for t in f["teachers"].values()]
        out.append(dict(f, teachers=teachers, docs=sorted(f["docs"])))
    return {"families": sorted(out, key=lambda f: f["family"])}


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
        from vqlab.agents.mcp import tools_runs as ms
        gpu = ms.t_gpu_state()
        mcp = ms.t_list_runs(40).get("runs", [])
    except Exception as e:
        gpu = {"error": f"{type(e).__name__}: {e}"}
    return {"runs": list(reversed(runs)), "mcp": mcp, "gpu": gpu}


def _tail(p, n=12, width=300):
    try:
        lines = [x[:width] for x in p.read_text(errors="replace").splitlines() if x.strip()]
    except OSError:
        return []
    return lines[-n:]


def _step_files(sd):
    """What a step left on disk: the command it ran and the end of its output
    (stderr, or stdout when stderr is empty) -- the 'why' behind a status."""
    if sd is None:
        return {"cmd": None, "tail": []}
    try:
        cmd = (sd / "cmd").read_text().strip()
    except OSError:
        cmd = None
    return {"cmd": cmd, "tail": _tail(sd / "stderr") or _tail(sd / "stdout")}


def queues_snapshot(n=40):
    """Most recent queue runs, newest first. `live` reuses the runner-alive
    check `vqlab queue status` makes (core.box.gpu_lease.pid_alive on state.pid)."""
    from vqlab.core.box.gpu_lease import pid_alive
    from vqlab.queue import files as rq
    dirs = sorted((p for p in rq.queues_dir().glob("*") if (p / "state.json").exists()),
                  key=lambda p: p.name, reverse=True)[:max(1, n)]
    out = []
    for d in dirs:
        try:
            s = json.loads((d / "state.json").read_text())
        except ValueError:
            continue
        sdirs = {p.name.split("-", 1)[0]: p for p in (d / "steps").glob("*") if p.is_dir()}
        out.append({"id": d.name, "name": s.get("name"), "status": s.get("status"),
                    "live": bool(pid_alive(s.get("pid"))), "preflight": bool(s.get("preflight")),
                    "created": s.get("created"), "host": s.get("host"), "commit": s.get("commit"),
                    "source": s.get("source"),
                    "steps": [dict({k: r.get(k) for k in ("name", "status", "seconds", "started", "finished",
                                                          "rc", "attempts", "reasons", "warnings")},
                                   **_step_files(sdirs.get(f"{i:02d}")))
                              for i, r in enumerate(s.get("steps", []))]})
    return out


def api_queues(q):
    return queues_snapshot(int(q.get("n", ["40"])[0]))


def api_settings(_q):
    from vqlab.agents import settings
    return {"config_file": str(config.config_file()),
            "config_exists": config.config_file().exists(),
            "locations": settings.locations(),
            "records": {"findings_log": str(config.findings_log()),
                        "registry": str(config.registry_dir()),
                        "families": str(config.families_dir()),
                        "research": str(config.research_dir()),
                        "queues": str(config.queues_dir())}}


def api_storage_usage(_q):
    from vqlab.agents import settings
    return settings.usage()


def api_scratch(_q):
    from vqlab.agents import settings
    return {"root": str(config.scratch()), "folders": settings.campaign_sizes()}


def api_scratch_reclaimable(_q):
    import shlex

    from vqlab.agents import scratch as scratch_cmd
    cands, kept = scratch_cmd.reclaimable([config.scratch()])
    return {"root": str(config.scratch()), "reclaimable": cands, "kept": kept,
            "total_bytes": sum(c["bytes"] for c in cands),
            "rm_lines": [f"rm -rf {shlex.quote(c['path'])}" for c in cands]}


def post_settings(body):
    from vqlab.agents import settings
    return settings.set_value(str(body.get("key", "")), str(body.get("path", "")),
                              create=bool(body.get("create")))


ROUTES = {"/api/fleet": api_fleet, "/api/provenance": api_provenance,
          "/api/fits": api_fits, "/api/families": api_families, "/api/runs": api_runs,
          "/api/queues": api_queues, "/api/settings": api_settings,
          "/api/scratch": api_scratch, "/api/storage/usage": api_storage_usage, "/api/scratch/reclaimable": api_scratch_reclaimable}
POSTS = {"/api/settings": post_settings}


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

    def _same_origin(self) -> bool:
        """A POST must come from this page: a JSON body, and an Origin (when
        the browser sends one) naming this server's own host:port."""
        if (self.headers.get("Content-Type") or "").split(";")[0].strip() != "application/json":
            return False
        origin = self.headers.get("Origin")
        host = self.headers.get("Host") or ""
        return origin is None or origin == f"http://{host}"

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        fn = POSTS.get(u.path)
        if fn is None:
            return self._send(404, b'{"error":"not found"}', "application/json")
        if not self._same_origin():
            return self._send(403, b'{"error":"cross-origin or non-JSON request refused"}',
                              "application/json")
        try:
            n = min(int(self.headers.get("Content-Length") or 0), 65536)
            body = json.loads(self.rfile.read(n) or b"{}")
            data = fn(body if isinstance(body, dict) else {})
            code = 200 if data.get("ok", True) else 400
        except Exception as e:
            data, code = {"ok": False, "error": f"{type(e).__name__}: {e}"}, 500
        self._send(code, json.dumps(data, default=str).encode(), "application/json")

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path in ("/", "/index.html"):            # the GUI: Library / Recipe / Jobs
            return self._bench("index.html")
        if u.path in ("/bench", "/bench/"):           # old address of the same page
            self.send_response(301)
            self.send_header("Location", "/")
            self.end_headers()
            return
        fn = ROUTES.get(u.path)
        if fn is None:
            return self._send(404, b'{"error":"not found"}', "application/json")
        try:
            data = fn(urllib.parse.parse_qs(u.query))
            code = 200
        except Exception as e:
            data, code = {"error": f"{type(e).__name__}: {e}"}, 500
        self._send(code, json.dumps(data, default=str).encode(), "application/json")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vqlab gui", description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=8781)
    ap.add_argument("--open", action="store_true", help="open it in the default browser")
    a = ap.parse_args(argv)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://127.0.0.1:{a.port}/"
    print(f"vqlab gui on {url}  (ctrl-c to stop)", flush=True)
    if a.open:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
