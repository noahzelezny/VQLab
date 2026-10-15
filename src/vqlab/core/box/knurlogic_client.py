"""Knurlogic HTTP/job client helpers shared by the bench (speed-pair-knurlogic)
and the gates (check-release --knurlogic-smoke, vision-smoke --knurlogic).

In core so gate does not import bench. Pure stdlib; the Knurlogic MCP module
itself (`K`) is always passed in by the caller.
"""
from __future__ import annotations

import json
import os
import pathlib
import urllib.request


def models_dir() -> pathlib.Path:
    """The Knurlogic models folder the gates look a name up in: KNURLOGIC_MODELS,
    else ~/.exo/models. check-release --knurlogic-smoke, vision-smoke --knurlogic
    and knurlogic-link all read it here, so the entry one makes is the entry the
    others check."""
    return pathlib.Path(os.environ.get("KNURLOGIC_MODELS", "~/.exo/models")).expanduser()


def post(url, doc, timeout=1800):
    base = url.rstrip("/")
    if not base.endswith("/v1"):          # Knurlogic's load() url already ends in /v1
        base += "/v1"
    req = urllib.request.Request(base + "/chat/completions",
                                 data=json.dumps(doc).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# Request fields that keep a reasoning model's answer in `content`: thinking
# on, a smoke's token budget goes to the reasoning channel and content is empty.
NO_THINKING = {"reasoning_effort": "none"}

BUNDLED = "ships its own runtime (model.py) and it WILL be executed"
VENDORED = "runtime serves it instead"


def rank_logs(job, hosts):
    """{machine: log text} for every rank of `job`. The rank log is the only
    authoritative record of which VQ runtime loaded (Knurlogic, 2026-09-29:
    SERVED["runtime"] is not exposed over HTTP; state()'s `runtime` is the
    server kind). `hosts` maps a machine name to "user@host" for ssh; the
    rest are read locally."""
    import glob
    import subprocess
    out = {}
    for machine, host in hosts.items():
        cmd = f"cat ~/.cache/knurlogic/jobs/{job}/rank*.log 2>/dev/null"
        if host:
            p = subprocess.run(["ssh", host, cmd], capture_output=True, text=True, timeout=60)
            out[machine] = p.stdout
        else:
            out[machine] = "".join(pathlib.Path(f).read_text() for f in
                                   glob.glob(str(pathlib.Path.home() /
                                                 f".cache/knurlogic/jobs/{job}/rank*.log")))
    return out


def ready(e):
    return e.get("phase") == "ready" or (not e.get("job") and e.get("state") == "loaded")


def entry(K, job):
    for m in K.state().get("models", []):
        if str(m.get("job")) == str(job) or str(m.get("instance")) == str(job):
            return m
    return None


def drafting(url):
    """Knurlogic's cumulative drafting counters from /status.json (None when
    the server does not report them). Deltas around one request give its
    acceptance."""
    root = url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    try:
        with urllib.request.urlopen(root + "/status.json", timeout=10) as r:
            return (json.loads(r.read()).get("drafting") or None)
    except Exception:
        return None
