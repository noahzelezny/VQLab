#!/usr/bin/env python3
"""vqlab reserve: say who has a box until when, where every box can see it.

    vqlab reserve m4 --for knurlogic --until +3h [--note "vision bench"]
    vqlab reserve here --for noah --until 02:30
    vqlab reserve --list                  # active reservations, every box
    vqlab reserve --release [m4|here]     # default: here

WHY (2026-10-03): the GPU lease is per box and says nothing about intent.
Two sessions negotiated "M4 free?" by message four times in one night. A
reservation is that intent, written down: `queue run` (and `queue run --on
BOX`) refuses a box reserved for someone else unless --override-reservation,
which the queue's state.json records; MCP `gpu_state` lists every box's.

WHERE. One JSON file, `<scratch>/vqlab-reservations.json` ($VQLAB_RESERVATIONS
overrides). Scratch is the shared SSD that every box mounts (the M3 owns the
Storage, the M4 reaches it over SMB, and `queue run --on` already relies on
that for its queue dir), so a reservation written on either box is read by
both. When scratch is not reachable (SSD unmounted), it falls back to
`~/.vqlab/reservations.json` on this host and SAYS so: such a reservation is
visible only here. Reads merge both files. Expired reservations are ignored
and pruned on the next write.

Who is "someone else": a reservation's `for` compared with $VQLAB_WHO, else
the login name.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import getpass
import json
import os
import pathlib
import re
import socket
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # src/
from vqlab import config  # noqa: E402

FILE = "vqlab-reservations.json"


def _host() -> str:
    return socket.gethostname().split(".")[0]


def here() -> str:
    """This box's name: its [boxes.NAME] entry when it has one, else the
    short hostname."""
    return config.this_box() or _host()


def who() -> str:
    return os.environ.get("VQLAB_WHO") or getpass.getuser()


def shared_path() -> pathlib.Path:
    env = os.environ.get("VQLAB_RESERVATIONS")
    return pathlib.Path(env) if env else config.scratch() / FILE


def local_path() -> pathlib.Path:
    return config.HOME / "reservations.json"


def _box(name: str) -> str:
    return here() if name in (None, "", "here") else name


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def parse_until(s: str, now: dt.datetime | None = None) -> dt.datetime:
    """'+3h', '+90m', '+2h30m', '+1d' | 'HH:MM' (local; tomorrow if already
    past) | an ISO time (local if it has no zone). Returns aware UTC."""
    now = now or _now()
    s = s.strip()
    if s.startswith("+"):
        parts = re.findall(r"(\d+(?:\.\d+)?)([dhm])", s[1:])
        if not parts or "".join(n + u for n, u in parts) != s[1:]:
            raise ValueError(f"bad duration {s!r}: use e.g. +3h, +90m, +2h30m, +1d")
        mult = {"d": 86400, "h": 3600, "m": 60}
        return now + dt.timedelta(seconds=sum(float(n) * mult[u] for n, u in parts))
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", s)
    loc = now.astimezone()
    if m:
        t = loc.replace(hour=int(m.group(1)), minute=int(m.group(2)), second=0, microsecond=0)
        if t <= loc:
            t += dt.timedelta(days=1)
        return t.astimezone(dt.timezone.utc)
    try:
        t = dt.datetime.fromisoformat(s)
    except ValueError:
        raise ValueError(f"bad time {s!r}: use +3h, HH:MM or an ISO time") from None
    if t.tzinfo is None:
        t = t.replace(tzinfo=loc.tzinfo)
    return t.astimezone(dt.timezone.utc)


def _read(p: pathlib.Path) -> list:
    try:
        return json.loads(p.read_text()).get("reservations", [])
    except (OSError, ValueError, AttributeError):
        return []


def _active(rs: list, now: dt.datetime | None = None) -> list:
    now = now or _now()
    out = []
    for r in rs:
        try:
            if dt.datetime.fromisoformat(r["until"]) > now:
                out.append(r)
        except (KeyError, ValueError, TypeError):
            continue
    return out


def active() -> list:
    """Every unexpired reservation, shared file first, then this host's
    fallback file (marked `local_only`)."""
    rs = [dict(r, store=str(shared_path())) for r in _active(_read(shared_path()))]
    seen = {r["box"] for r in rs}
    for r in _active(_read(local_path())):
        if r.get("box") not in seen:
            rs.append(dict(r, store=str(local_path()), local_only=True))
    return rs


def get(box: str) -> dict | None:
    box = _box(box)
    return next((r for r in active() if r.get("box") == box), None)


def _writable_store() -> tuple[pathlib.Path, bool]:
    """(path, shared?) -- the shared file when its directory is reachable."""
    p = shared_path()
    try:
        if p.parent.is_dir() and os.access(p.parent, os.W_OK):
            return p, True
    except OSError:
        pass
    return local_path(), False


@contextlib.contextmanager
def _locked(p: pathlib.Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = None
    try:
        import fcntl
        fd = os.open(str(p) + ".lock", os.O_CREAT | os.O_RDWR, 0o644)
        fcntl.flock(fd, fcntl.LOCK_EX)
    except OSError:
        fd = None              # SMB without flock: the atomic replace still holds
    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)


def _write(p: pathlib.Path, rs: list) -> None:
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps({"schema": "vqlab.reservations/1", "reservations": rs}, indent=1))
    tmp.replace(p)


def reserve(box: str, for_: str, until: str, note: str = "", force: bool = False) -> dict:
    box = _box(box)
    t = parse_until(until)
    if t <= _now():
        raise SystemExit(f"--until {until} is in the past")
    cur = get(box)
    if cur and cur.get("for") != for_ and not force:
        raise SystemExit(f"{box} is reserved for {cur['for']} until {_fmt(cur['until'])}"
                         + (f" ({cur['note']})" if cur.get("note") else "")
                         + "; --force to replace it, or ask them")
    p, shared = _writable_store()
    rec = {"box": box, "for": for_, "until": t.isoformat(timespec="seconds"),
           "note": note, "by": f"{who()}@{_host()}", "created": _now().isoformat(timespec="seconds")}
    with _locked(p):
        rs = [r for r in _active(_read(p)) if r.get("box") != box] + [rec]
        _write(p, rs)
    if not shared:
        print(f"WARNING: {shared_path().parent} is not reachable; this reservation is in "
              f"{p} and visible ONLY on {_host()}", file=sys.stderr)
    return dict(rec, store=str(p))


def release(box: str) -> list:
    box = _box(box)
    gone = []
    for p in (shared_path(), local_path()):
        if not p.exists():
            continue
        try:
            with _locked(p):
                rs = _active(_read(p))
                gone += [r for r in rs if r.get("box") == box]
                _write(p, [r for r in rs if r.get("box") != box])
        except OSError as e:
            print(f"could not update {p}: {e}", file=sys.stderr)
    return gone


def _fmt(iso: str) -> str:
    t = dt.datetime.fromisoformat(iso).astimezone()
    left = (t - _now()).total_seconds()
    return f"{t:%Y-%m-%d %H:%M %Z} (in {_dur(left)})"


def _dur(s: float) -> str:
    s = max(0, int(s))
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m"


def describe(r: dict) -> str:
    return (f"{r['box']}: reserved for {r['for']} until {_fmt(r['until'])}"
            + (f" -- {r['note']}" if r.get("note") else "")
            + f"  [by {r.get('by', '?')}{', LOCAL ONLY' if r.get('local_only') else ''}]")


def refusal(box: str, override: bool = False) -> str | None:
    """None when this caller may use `box`; else the plain reason it may not.
    A reservation for the caller ($VQLAB_WHO / login) is not a refusal."""
    r = get(box)
    if r is None and box in (None, "", "here"):
        r = get(_host())              # reserved by hostname from a box that has no name for this one
    if not r or r.get("for") == who() or override:
        return None
    return (f"REFUSED: {describe(r)}. You are {who()} ($VQLAB_WHO). Ask them, wait, or pass "
            f"--override-reservation (recorded in the queue's state.json).")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vqlab reserve", description=__doc__.split("\n")[0])
    ap.add_argument("box", nargs="?", help="a [boxes.NAME], or `here`")
    ap.add_argument("--for", dest="for_", help="who has it (a session, a person)")
    ap.add_argument("--until", help="+3h, +90m, HH:MM (local), or an ISO time")
    ap.add_argument("--note", default="")
    ap.add_argument("--force", action="store_true", help="replace someone else's reservation")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--release", action="store_true", help="drop the box's reservation (default here)")
    a = ap.parse_args(argv)
    if a.list:
        rs = active()
        print(f"reservations ({shared_path()}; this host's fallback {local_path()}):")
        for r in rs:
            print("  " + describe(r))
        if not rs:
            print("  none active")
        return 0
    if a.release:
        gone = release(a.box)
        print(f"released {_box(a.box)}" if gone else f"{_box(a.box)} had no active reservation")
        return 0
    if not (a.box and a.for_ and a.until):
        ap.error("give BOX --for WHO --until TIME, or --list, or --release")
    try:
        r = reserve(a.box, a.for_, a.until, a.note, a.force)
    except ValueError as e:
        ap.error(str(e))
    print("reserved " + describe(r) + f"\n  -> {r['store']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
