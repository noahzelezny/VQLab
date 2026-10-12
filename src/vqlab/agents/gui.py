#!/usr/bin/env python3
"""vqlab gui — the VQ Lab GUI: Library · Recipe · Jobs.

    vqlab gui [--port 8781] [--open]

Serves gui_static/index.html at / on 127.0.0.1 (never another
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
  POST /api/pick-folder  the system's own folder chooser -> {ok, path}
  /api/models/found   models already on this machine (vqlab lab scan; rescanned at
                    launch and by POST /api/models/scan)

  /api/teachers      profiled teachers (sizes per geometry, fits, buildable) and
                    scanned originals not yet profiled; POST /profile profiles one
  /api/scores        measured KL cells, newest per artifact x cache (records/scores.py)
  /api/downloads     Hub downloads into <teachers>; POST /start, /stop (agents/downloads.py)
  /api/jobs/files    queue files the Jobs page can start (<records>/queue-files)
  POST /api/jobs/recipe  a Recipe -> a queue file (plan/recipe_queue.py); /preview
                    the steps it would compile to, written nowhere
  POST /api/jobs/plan|start|stop|resume  a queue run (agents/jobs.py)

Stdlib only; reads index files and small JSON, never a tensor, so it is safe
on a box that is mid-experiment. POSTs are same-origin JSON only, on
127.0.0.1: /api/settings writes one key of the user's config file,
/api/models/scan rescans, /api/pick-folder opens the folder chooser, and
/api/jobs/* start, stop or resume a queue through `vqlab queue run` (so its
pinned tree, GPU lease and exo gate apply); a start needs the token of the
plan the operator confirmed. Nothing on the page publishes, pushes or
deletes.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import pathlib
import sys
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


from vqlab import config
from vqlab.agents import downloads, jobs, teachers

PAGE = pathlib.Path(__file__).resolve().parent / "gui_static" / "index.html"


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
    from vqlab.gate import check_bundle
    arts = _jsonl(config.registry_dir() / "artifacts.jsonl")
    hub = {h["repo"]: h for h in _jsonl(config.registry_dir() / "hub.jsonl")}
    twins = {}
    for e in arts:
        twins.setdefault(e["fingerprint"], []).append(e["name"])
    for e in arts:
        e["hub"] = hub.get(e.get("hf_repo") or "")
        e["same_bytes_as"] = [n for n in twins[e["fingerprint"]] if n != e["name"]]
        e["bundle"] = check_bundle.verdict(e["path"])
    return {"artifacts": arts}


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
    from vqlab.queue import eta as queue_eta
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
        live = bool(pid_alive(s.get("pid")))
        out.append({"id": d.name, "name": s.get("name"), "status": s.get("status"), "live": live,
                    "eta": queue_eta.queue_eta(d, rq.queues_dir()) if live else None,
                    "preflight": bool(s.get("preflight")),
                    "created": s.get("created"), "host": s.get("host"), "commit": s.get("commit"),
                    "source": s.get("source"),
                    "steps": [dict({k: r.get(k) for k in ("name", "status", "seconds", "started", "finished",
                                                          "rc", "attempts", "reasons", "warnings")},
                                   **_step_files(sdirs.get(f"{i:02d}")))
                              for i, r in enumerate(s.get("steps", []))]})
    return out


def api_queues(q):
    return queues_snapshot(int(q.get("n", ["40"])[0]))


def api_scores(_q):
    from vqlab.records import scores
    return scores.latest()


def api_downloads(_q):
    return downloads.downloads()


def api_teachers(_q):
    return teachers.teachers()


def api_jobs_files(_q):
    return jobs.queue_files()


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


def api_models_found(_q):
    from vqlab.agents import model_scan
    rows = model_scan.load()
    return {"models": rows, "scanning": _SCAN_LOCK.locked(),
            "scanned": rows[0].get("scanned") if rows else None}


_SCAN_LOCK = threading.Lock()      # held while a scan runs


def _scan_in_background():
    """Scan for models without holding up the page (launch and Rescan); a
    scan already running is not started twice."""
    if not _SCAN_LOCK.acquire(blocking=False):
        return

    def go():
        try:
            from vqlab.agents import model_scan
            model_scan.refresh()
        except Exception as e:  # report, never kill the server
            print(f"vqlab gui: model scan failed: {type(e).__name__}: {e}", file=sys.stderr)
        finally:
            _SCAN_LOCK.release()
    threading.Thread(target=go, daemon=True).start()


def post_models_scan(_body):
    _scan_in_background()
    return {"ok": True, "scanning": True}


_PROMPT = "Choose a folder for VQ Lab"
# Each chooser gets the start folder as an ARGUMENT (argv / env), never
# spliced into its script text, so no path can break or inject into it.
_OSA_PICK = ("on run argv\n"
             f'  set p to "{_PROMPT}"\n'
             "  if (count of argv) > 0 then\n"
             "    return POSIX path of (choose folder with prompt p"
             " default location (POSIX file (item 1 of argv)))\n"
             "  end if\n"
             "  return POSIX path of (choose folder with prompt p)\n"
             "end run")
_PS_PICK = ("Add-Type -AssemblyName System.Windows.Forms;"
            "$d=New-Object System.Windows.Forms.FolderBrowserDialog;"
            "$d.SelectedPath=$env:VQLAB_PICK_START;"
            "if($d.ShowDialog() -eq 'OK'){$d.SelectedPath}")
_TK_PICK = ("import sys, tkinter as t, tkinter.filedialog as f\n"
            "r = t.Tk(); r.withdraw(); r.attributes('-topmost', True)\n"
            f"print(f.askdirectory(initialdir=sys.argv[1] or None, title={_PROMPT!r}) or '')")


def _pick_commands(start: str) -> list[list[str]]:
    """Folder-chooser commands for this OS, best first. Each prints the
    chosen path; an empty result or a cancel message means cancelled."""
    import shutil
    if sys.platform == "darwin":
        return [["osascript", "-e", _OSA_PICK, *([start] if start else [])]]
    out = []
    if sys.platform.startswith("win"):
        out.append(["powershell", "-NoProfile", "-STA", "-Command", _PS_PICK])
    else:
        if shutil.which("zenity"):
            out.append(["zenity", "--file-selection", "--directory", f"--title={_PROMPT}",
                        f"--filename={start}/"])
        if shutil.which("kdialog"):
            out.append(["kdialog", "--getexistingdirectory", start or "."])
    out.append([sys.executable, "-c", _TK_PICK, start])      # Tk: any OS with a display
    return out


def post_pick_folder(body):
    """The system's own folder chooser (macOS chooser, zenity / kdialog,
    Windows' dialog, else Tk), starting at body['start'] or its nearest
    existing parent. Returns {ok, path} or {ok: False, cancelled|error}. The
    server listens on 127.0.0.1 only, so the dialog opens for the person at
    this machine."""
    import subprocess
    start = pathlib.Path(str(body.get("start") or "")).expanduser()
    while start != start.parent and not start.is_dir():
        start = start.parent
    s = str(start) if start.is_dir() and start != start.parent else ""
    errors = []
    for cmd in _pick_commands(s):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                               env={**os.environ, "VQLAB_PICK_START": s})
        except FileNotFoundError:
            continue                                  # not installed: try the next
        except (OSError, subprocess.TimeoutExpired) as e:
            errors.append(f"{type(e).__name__}: {e}")
            continue
        path = r.stdout.strip()
        if r.returncode == 0 and path:
            return {"ok": True, "path": path.rstrip("/\\") or path}
        err = r.stderr.strip()
        # Closing a chooser: an empty result with exit 0 or 1 (zenity and
        # kdialog print unrelated toolkit warnings to stderr, so stderr alone
        # does not mean failure), or osascript's "User canceled (-128)".
        if r.returncode in (0, 1) and (cmd[0] in ("zenity", "kdialog") or not err
                                       or "cancel" in err.lower() or "-128" in err):
            return {"ok": False, "cancelled": True}
        errors.append(err[:200])
    return {"ok": False, "error": "no folder chooser available here; type the path"
            + (f" ({errors[-1]})" if errors else "")}


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
          "/api/scratch": api_scratch, "/api/storage/usage": api_storage_usage,
          "/api/models/found": api_models_found,
          "/api/jobs/files": api_jobs_files, "/api/teachers": api_teachers,
          "/api/downloads": api_downloads, "/api/scores": api_scores,
          "/api/scratch/reclaimable": api_scratch_reclaimable}
POSTS = {"/api/settings": post_settings, "/api/models/scan": post_models_scan,
         "/api/pick-folder": post_pick_folder, "/api/jobs/plan": jobs.plan,
         "/api/jobs/start": jobs.start, "/api/jobs/stop": jobs.stop, "/api/jobs/resume": jobs.resume,
         "/api/jobs/recipe": jobs.write_recipe, "/api/jobs/preview": jobs.preview,
         "/api/teachers/profile": teachers.profile,
         "/api/downloads/start": downloads.start, "/api/downloads/stop": downloads.stop}


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

    def _local_host(self) -> bool:
        """The Host header must name this machine at the bound port. Without
        this, DNS rebinding (a site whose name resolves to 127.0.0.1) is
        "same origin" with itself and could read every endpoint and POST."""
        port = self.server.server_address[1]
        return (self.headers.get("Host") or "").lower() in {
            f"{h}:{port}" for h in ("127.0.0.1", "localhost", "[::1]")}

    def _refuse_foreign_host(self) -> bool:
        if self._local_host():
            return False
        self._send(403, b'{"error":"requests must address this machine (127.0.0.1)"}',
                   "application/json")
        return True

    def _same_origin(self) -> bool:
        """A POST must come from this page: a JSON body, and an Origin (when
        the browser sends one) naming this server's own host:port."""
        if (self.headers.get("Content-Type") or "").split(";")[0].strip() != "application/json":
            return False
        origin = self.headers.get("Origin")
        host = self.headers.get("Host") or ""
        return origin is None or origin == f"http://{host}"

    def do_POST(self):
        if self._refuse_foreign_host():
            return
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
        if self._refuse_foreign_host():
            return
        u = urllib.parse.urlparse(self.path)
        if u.path in ("/", "/index.html"):            # the GUI: Library / Recipe / Jobs
            return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
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
    _scan_in_background()          # models already on this machine (vqlab lab scan)
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
