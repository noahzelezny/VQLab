"""Knurlogic HTTP/job client helpers shared by the bench (speed-pair-knurlogic)
and the gates (check-release --knurlogic-smoke, vision-smoke --knurlogic).

In core so gate does not import bench. Pure stdlib; the Knurlogic MCP module
itself (`K`) is always passed in by the caller.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time
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


# Refusals for room that a just-unloaded job is still handing back: a Mac
# frees an unloaded model's memory some time AFTER its job is gone, so a load
# right behind an unload is refused for memory about to return.
ROOM_REFUSALS = ("cannot place", "fits on no set")


def mcp(src: str = ""):
    """Knurlogic's MCP module, with `src` (default $KNURLOGIC_SRC), a Knurlogic
    source tree, first on the path when given; None when not importable."""
    ksrc = src or os.environ.get("KNURLOGIC_SRC", "")
    if ksrc and ksrc not in sys.path:
        sys.path.insert(0, ksrc)
    try:
        from knurlogic.interfaces import mcp as K
    except ImportError:
        return None
    return K


def linked_name(artifact: pathlib.Path) -> str:
    """The name Knurlogic loads `artifact` by. Raises ValueError when the
    models-dir entry of that name resolves anywhere else: the smoke would load
    something other than this artifact."""
    name = artifact.resolve().name
    entry_path = models_dir() / name
    if entry_path.resolve() != artifact.resolve():
        raise ValueError(f"Knurlogic models dir entry {entry_path} does not resolve to "
                         f"{artifact.resolve()}: the smoke would load something else")
    return name


def load(K, name: str, *, machines, split=None, link=None, sets=None,
         tries: int = 10, wait_s: float = 30, sleep=time.sleep) -> dict:
    """K.load, retrying only refusals for room still being freed
    (ROOM_REFUSALS), `tries` loads in all; the last answer is returned."""
    for _ in range(tries - 1):
        out = K.load(artifact=name, machines=machines, split=split, link=link, sets=sets)
        refused = str(out.get("refused") or "")
        if not any(r in refused for r in ROOM_REFUSALS):
            return out
        print(f"  {name}: placement refused ({refused}); memory still returning, "
              f"retry in {wait_s:g} s", file=sys.stderr)
        sleep(wait_s)
    return K.load(artifact=name, machines=machines, split=split, link=link, sets=sets)


def unload(K, out: dict, *, polls: int = 60, wait_s: float = 5, sleep=time.sleep) -> None:
    """Unload what load() returned (a pipeline job, or a single Mac's instance
    when its job is null) and wait until it is gone everywhere."""
    job = out.get("job")
    K.unload(**({"job": str(job)} if job else {"instance": str(out.get("instance"))}))
    for _ in range(polls):
        if entry(K, job or out.get("instance")) is None:
            return
        sleep(wait_s)
