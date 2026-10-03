#!/usr/bin/env python3
"""vqlab config: show where vqlab reads and writes, or write a starter config.

    vqlab config                      # every location, and where it came from
    vqlab config init --root DIR      # write ~/.config/vqlab/config.toml under DIR
    vqlab config init --scratch A --models B --teachers C --fit-store D

Nothing in vqlab assumes a disk layout: each location resolves from its
environment variable, then the config file, then a default under ~/.vqlab
(src/vqlab/config.py). Large models do not belong on a small system disk, so
point these at a big volume before the first fit.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

from vqlab import config as C


def _source(key):
    env = C._ENV[key]
    if os.environ.get(env):
        return f"${env}"
    if key == "teachers" and C.box_teachers():
        return f"[boxes.{C.this_box()}] teachers: running on that box"
    if C._file_paths(str(C.config_file())).get(key):
        return str(C.config_file())
    return "default"


def show():
    print(f"config file: {C.config_file()}"
          + ("" if C.config_file().exists() else "  (absent: defaults under ~/.vqlab)"))
    rows = [("scratch", C.scratch()), ("models", C.models()), ("teachers", C.teachers()),
            ("fit_store", ", ".join(map(str, C.fit_store()))), ("gpu_lease", C.gpu_lease())]
    for k, v in rows:
        print(f"  {k:10s} {str(v):60s} [{_source(k)}]")
    print(f"  {'roots':10s} {', '.join(map(str, C.roots()))}")
    for name, b in C.boxes().items():
        print(f"  box {name}: ssh {b.get('ssh')}, repo {b.get('repo')}")
        print("      teachers " + (f"{b['teachers']}  (local copy; used when running on {name})"
                                   if b.get("teachers") else
                                   f"= [paths] teachers (add `teachers = \"<path>\"` under "
                                   f"[boxes.{name}] to use a local copy there)"))
    print(f"  this box: {C.this_box() or 'home (no [boxes.*] entry matches)'}"
          f"  [$VQLAB_BOX, else hostname vs box name / `hostname`]")
    return 0


def init(a):
    path = C.config_file()
    if path.exists() and not a.force:
        raise SystemExit(f"{path} exists; --force to overwrite (or edit it)")
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    for k, v in loc.items():
        pathlib.Path(v).expanduser().mkdir(parents=True, exist_ok=True)
    C._file_paths.cache_clear()
    print(f"wrote {path}")
    return show()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vqlab config", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd")
    pi = sub.add_parser("init", help="write a starter config file")
    pi.add_argument("--root", help="one directory holding scratch/ models/ teachers/ fits/")
    for k in ("scratch", "models", "teachers", "fit-store"):
        pi.add_argument(f"--{k}")
    pi.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    return init(a) if a.cmd == "init" else show()


if __name__ == "__main__":
    sys.exit(main())
