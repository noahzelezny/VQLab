"""Refuse to publish text that names this machine.

Build records, cards and bundled code are uploaded as they sit on disk, and
on 2026-10-10 the records alone carried ~41 absolute local paths per rung
(the checkout, scratch, the models root). This scans every text file about
to ship for:

  * the home directory of the user running it,
  * every configured storage root and the checkout (`config.private_prefixes`),
  * a private (RFC 1918) IPv4 address,
  * this machine's hostname and every configured box name / host,

all read from config / the environment at runtime, never written here.
`vqlab provenance --portable <artifact>` rewrites old build records; any
other file is fixed by hand.
"""
from __future__ import annotations

import pathlib
import re

from vqlab import config
# Local backup / working copies: never diffed against the Hub, never in a
# publish line, never uploaded (one definition, in core/artifacts/hub.py).
from vqlab.core.artifacts.hub import is_backup  # noqa: F401

RECORD_HOST = re.compile(r'"host"\s*:\s*"[^"]+"')
TEXT_SUFFIXES = {".json", ".jsonl", ".md", ".py", ".txt", ".jinja"}
LAN_IP = re.compile(
    r"(?<![\d.])(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}(?![\d.])")


def _needles() -> list[tuple[str, str]]:
    return [(pre, f"local path ({label})") for pre, label in config.private_prefixes()]


def host_names() -> set[str]:
    """This machine's hostname (full and short, from socket.gethostname())
    and every configured box: its name, `hostname` and ssh host. Read at
    runtime; none is written in code."""
    import socket
    out = set()
    try:
        h = socket.gethostname()
        out |= {h, h.split(".")[0]}
    except OSError:
        pass
    try:
        for name, b in (config.boxes() or {}).items():
            out.add(name)
            if isinstance(b, dict):
                for k in ("hostname",):
                    if b.get(k):
                        out.add(str(b[k]))
                if b.get("ssh"):
                    out.add(str(b["ssh"]).split("@")[-1].split(":")[0])
    except Exception:
        pass
    return {h for h in out if h and len(h) >= 2 and h.lower() != "localhost"}


def _host_re(hosts):
    if not hosts:
        return None
    alt = "|".join(re.escape(h) for h in sorted(hosts, key=len, reverse=True))
    return re.compile(r"(?<![\w.-])(?:" + alt + r")(?![\w-])")


def scan_file(p: pathlib.Path, needles=None, hosts=None) -> list[tuple[int, str]]:
    """[(line number, what)] for every leak in one text file."""
    needles = needles if needles is not None else _needles()
    hre = _host_re(hosts if hosts is not None else host_names())
    if p.suffix.lower() not in TEXT_SUFFIXES:
        return []
    try:
        text = p.read_text(errors="replace")
    except OSError:
        return []
    hits = []
    for n, line in enumerate(text.splitlines(), 1):
        for pre, what in needles:
            if re.search(re.escape(pre) + r"(?=/|$|[^\w.-])", line):
                hits.append((n, what))
                break
        else:
            m = LAN_IP.search(line)
            if m:
                hits.append((n, f"private LAN address {m.group(0)}"))
            elif hre is not None and (m := hre.search(line)):
                hits.append((n, f"machine name {m.group(0)!r}"))
            elif p.name.startswith("vqlab_provenance") and RECORD_HOST.search(line):
                # a build record's env.host names whichever machine built it,
                # including ones renamed since (gethostname cannot see those)
                hits.append((n, "build record env.host"))
    return hits


def scan(files, base: pathlib.Path) -> list[tuple[str, int, str]]:
    """[(relative file, line, what)] over `files` (paths under `base`)."""
    needles, hosts = _needles(), host_names()
    out = []
    for p in files:
        p = pathlib.Path(p)
        for n, what in scan_file(p, needles, hosts):
            out.append((str(p.relative_to(base)) if base in p.parents else str(p), n, what))
    return out


def report(hits, art) -> str:
    L = [f"   {f}:{n}: {what}" for f, n, what in hits[:20]]
    if len(hits) > 20:
        L.append(f"   ... and {len(hits) - 20} more")
    L.append(f"Build records: run `vqlab provenance --portable {art}` (try --dry-run "
             f"first). Any other file: remove the path by hand.")
    return "\n".join(L)


def scan_hub(repo: str, revision: str | None = None) -> tuple[list, int]:
    """Read-only: the same scan over a Hub repo's CURRENT text files (to
    verify after an upload). -> ([(file, line, what)], files scanned).
    Downloads text files only, into a temporary directory removed after."""
    import tempfile
    from huggingface_hub import HfApi, hf_hub_download
    names = [f for f in HfApi().list_repo_files(repo, revision=revision)
             if pathlib.Path(f).suffix.lower() in TEXT_SUFFIXES]
    with tempfile.TemporaryDirectory() as td:
        base = pathlib.Path(td).resolve()
        paths = [pathlib.Path(hf_hub_download(repo, f, revision=revision, local_dir=td)).resolve()
                 for f in names]
        hits = scan(paths, base)
    return hits, len(names)
