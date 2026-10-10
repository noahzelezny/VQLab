#!/usr/bin/env python3
"""vqlab config: show where vqlab reads and writes, set a location, or write a starter config.

    vqlab config                      # every location, its source and free space
    vqlab config --sizes              # + the size of each top-level scratch folder
    vqlab config set KEY PATH [--create]   # validate PATH, then write KEY
    vqlab config init                 # asks for every key in turn (in a terminal)
    vqlab config init --root DIR      # write ~/.config/vqlab/config.toml under DIR
    vqlab config init --scratch A --models B --teachers C --fit-store D [--records E]

Nothing in vqlab assumes a disk layout: each location resolves from its
environment variable, then the config file, then a default
(src/vqlab/config.py). Large models do not belong on a small system disk, so
point these at a big volume before the first fit. Validation and writing live
in agents/settings.py, shared with the GUI's Settings view.

An optional [card] table sets the support block `vqlab card` appends to model
cards (keys: sponsor_url, contact, blurb, contract; unset = no block). See
src/vqlab/config.py for an example.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

from vqlab import config as C
from vqlab.agents import settings as S


def show(sizes=False):
    print(f"config file: {C.config_file()}"
          + ("" if C.config_file().exists() else "  (absent: defaults)"))
    for r in S.locations():
        for i, (p, ok, free) in enumerate(zip(r["paths"], r["exists"], r["free_bytes"])):
            k = r["key"] if i == 0 else ""
            state = f"{S.gib(free)} free" if ok else f"missing ({S.gib(free)} free on its volume)"
            print(f"  {k:10s} {p:60s} {state:30s} {'[' + r['source'] + ']' if i == 0 else ''}")
    print(f"  records holds: findings log {C.findings_log()}, registry/, "
          f"families {C.families_dir()}, research/, queues {C.queues_dir()}")
    for name, b in C.boxes().items():
        print(f"  box {name}: ssh {b.get('ssh')}, repo {b.get('repo')}")
        print("      teachers " + (f"{b['teachers']}  (local copy; used when running on {name})"
                                   if b.get("teachers") else
                                   f"= [paths] teachers (add `teachers = \"<path>\"` under "
                                   f"[boxes.{name}] to use a local copy there)"))
    print(f"  this box: {C.this_box() or 'home (no [boxes.*] entry matches)'}"
          f"  [$VQLAB_BOX, else hostname vs box name / `hostname`]")
    card = C.card()
    print("  card      " + (", ".join(f"{k}={v!r}" for k, v in card.items()) if card else
                            "no [card] table: model cards get no support block "
                            "(keys: sponsor_url, contact, blurb, contract)"))
    if sizes:
        rows = S.campaign_sizes()
        print(f"scratch folders ({C.scratch()}):" + ("" if rows else " none"))
        for r in sorted(rows, key=lambda r: -(r["bytes"] or 0)):
            print(f"  {S.gib(r['bytes']):>11s}  {r['name']}"
                  + ("   (du timed out)" if r["bytes"] is None else ""))
        print("  (`vqlab scratch reclaimable` prints what is safe to delete)")
    else:
        print("  (--sizes: the size of each top-level scratch folder)")
    return 0


def cmd_set(a):
    res = S.set_value(a.key, a.path, create=a.create)
    if not res["ok"]:
        print(f"REFUSED: {a.key} = {a.path}: {res['error']}", file=sys.stderr)
        return 2
    print(f"{a.key} = {res['path']}  ({S.gib(res['free_bytes'])} free"
          + (", created" if res["created"] else "") + f") -> {res['config']}")
    if res.get("shadowed_by"):
        print(f"  note: {res['shadowed_by']} is set and overrides the file in this shell")
    return 0


def walk():
    """Ask for every key in turn, validating each answer (enter keeps the
    shown value; a missing directory is created)."""
    print(f"vqlab storage setup -> {C.config_file()}\n"
          "Large outputs (scratch, models, teachers, fits) belong on a big volume.")
    for r in S.locations():
        k = r["key"]
        if k == "roots":
            continue                              # derived from the others unless set
        cur = r["paths"][0]
        while True:
            ans = input(f"\n{k}: {r['description']}\n  [{cur}] "
                        f"({S.gib(r['free_bytes'][0])} free): ").strip()
            res = S.set_value(k, ans or cur, create=True)
            if res["ok"]:
                print(f"  ok: {res['path']} ({S.gib(res['free_bytes'])} free)")
                break
            print(f"  {res['error']}")
    return show()


def init(a):
    path = C.config_file()
    if path.exists() and not a.force:
        raise SystemExit(f"{path} exists; --force to overwrite (or `vqlab config set KEY PATH`)")
    if not (a.root or a.scratch or a.models or a.teachers or a.fit_store or a.records):
        if not sys.stdin.isatty():
            raise SystemExit("give --root (or every location), or run in a terminal to be asked")
        if path.exists():
            path.unlink()
        S._reset()
        return walk()
    base = pathlib.Path(a.root).expanduser() if a.root else None
    loc = {
        "scratch": a.scratch or (base and base / "scratch"),
        "models": a.models or (base and base / "models"),
        "teachers": a.teachers or (base and base / "teachers"),
        "fit_store": a.fit_store or (base and base / "fits"),
    }
    missing = [k for k, v in loc.items() if not v]
    if missing:
        raise SystemExit(f"give --root, or every one of --{', --'.join(m.replace('_', '-') for m in missing)}")
    lines = ["# vqlab storage (src/vqlab/config.py). Large outputs go only under these.", "[paths]"]
    for k in ("scratch", "models", "teachers"):
        lines.append(f'{k:9s} = "{pathlib.Path(loc[k]).expanduser()}"')
    lines.append(f'fit_store = ["{pathlib.Path(loc["fit_store"]).expanduser()}"]')
    rec = a.records or (base and base / "records")
    if rec:
        lines.append(f'records   = "{pathlib.Path(rec).expanduser()}"')
        loc["records"] = rec
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    for k, v in loc.items():
        pathlib.Path(v).expanduser().mkdir(parents=True, exist_ok=True)
    S._reset()
    print(f"wrote {path}")
    return show()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vqlab config", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", action="store_true",
                    help="also size each top-level scratch folder (du, time-capped)")
    sub = ap.add_subparsers(dest="cmd")
    pi = sub.add_parser("init", help="write a starter config file (no flags: asks for each key)")
    pi.add_argument("--root", help="one directory holding scratch/ models/ teachers/ fits/ records/")
    for k in ("scratch", "models", "teachers", "fit-store", "records"):
        pi.add_argument(f"--{k}")
    pi.add_argument("--force", action="store_true")
    ps = sub.add_parser("set", help="validate PATH and write it as KEY in the config file")
    ps.add_argument("key", choices=C.KEYS)
    ps.add_argument("path")
    ps.add_argument("--create", action="store_true", help="create the directory if it is missing")
    a = ap.parse_args(argv)
    if a.cmd == "init":
        return init(a)
    if a.cmd == "set":
        return cmd_set(a)
    return show(sizes=a.sizes)


if __name__ == "__main__":
    sys.exit(main())
