#!/usr/bin/env python3
"""knurlogic-link: make an artifact loadable by NAME through Knurlogic, without a copy.

    vqlab knurlogic-link <artifact>

Knurlogic loads by name: each Mac looks the name up in its model folders,
and `check-release --knurlogic-smoke` / `vision-smoke --knurlogic` refuse
unless the models-dir entry <models dir>/<artifact name> resolves to THIS
artifact. A build in scratch has no such entry. This makes it: ONE directory
symlink, so the entry resolves to the artifact itself and the gates'
identity check holds (a tree of per-file links would resolve to itself and
fail it). The models dir is the gates' own (KNURLOGIC_MODELS, else
~/.exo/models: core/box/knurlogic_client.models_dir).

Nothing is copied and nothing is deleted. An existing entry is replaced only
when it is a SYMLINK that dangles or points into the queue-preflight area (a
preflight's output, removed when the preflight passed); a real folder, or a
link to anything else, is refused by name: removing it is a human's call.

A pipeline peer resolves the link on its own side, so the target path must be
the same on every Mac that serves a rank (the shared volume mounted under one
name, as the queue's pins already rely on).
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

from vqlab import config
from vqlab.core.box.knurlogic_client import models_dir


def _preflight_area() -> pathlib.Path:
    """Where queue preflights write (queue/steps.preflight_root, which gate may not import)."""
    env = os.environ.get("VQLAB_PREFLIGHT_DIR")
    return pathlib.Path(env) if env else config.scratch() / "queue-preflight"


def _replaceable(entry: pathlib.Path) -> bool:
    if not entry.is_symlink():
        return False
    target = pathlib.Path(os.path.realpath(entry))
    if not target.exists():
        return True
    try:
        target.relative_to(_preflight_area().resolve())
        return True
    except (ValueError, OSError):
        return False


def link(artifact, md: pathlib.Path | None = None) -> tuple[int, str]:
    """-> (exit code, message)."""
    art = pathlib.Path(artifact).resolve()
    if not (art / "config.json").is_file():
        return 2, f"FAIL: {art} is not an artifact (no config.json)"
    md = md if md is not None else models_dir()
    if not md.is_dir():
        return 2, (f"FAIL: the Knurlogic models dir {md} does not exist (set KNURLOGIC_MODELS "
                   "to the folder Knurlogic serves from)")
    entry = md / art.name
    if entry.exists() or entry.is_symlink():
        if entry.exists() and entry.resolve() == art:
            return 0, f"OK: {entry} already resolves to {art}"
        if not _replaceable(entry):
            what = "a real folder" if not entry.is_symlink() else f"a link to {os.readlink(entry)}"
            return 2, (f"FAIL: {entry} exists and is {what}, not this artifact ({art}). "
                       "Knurlogic would load that instead; move or remove it by hand, "
                       "or rename the build")
        old = os.readlink(entry)
        entry.unlink()          # a symlink: removes the link, never data
        print(f"replaced stale link {entry} -> {old}")
    entry.symlink_to(art, target_is_directory=True)
    if entry.resolve() != art:
        return 2, f"FAIL: {entry} was linked but resolves to {entry.resolve()}, not {art}"
    return 0, (f"OK: {entry} -> {art}\n"
               "  every Mac serving a rank must reach that path under the same name")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vqlab knurlogic-link", description=__doc__.split("\n")[0])
    ap.add_argument("artifact")
    a = ap.parse_args(argv)
    rc, msg = link(a.artifact)
    print(msg)
    return rc


if __name__ == "__main__":
    sys.exit(main())
