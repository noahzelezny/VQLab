#!/usr/bin/env python3
"""HARD GUARD: refuse any resident-memory operation on a model bigger than RAM.

WHY THIS IS A SCRIPT AND NOT A RULE. The rule already existed (FINDINGS III.4
for speed tests, III.11a for smokes) and was violated THREE TIMES on 2026-08-21
by the same agent on the same box: a proposed d8 speed test, then a d8 smoke,
then a 143.682 GiB smoke on a 96 GiB machine that drove swap to 60 of 61 GiB.
A rule that gets restated after each violation is not a control. This is.

    vqlab preflight-ram <artifact-dir> [--headroom 0.90]

Exit 0 if the artifact's safetensors fit in physical RAM with headroom.
Exit 2 otherwise, naming the box, the artifact and where it CAN run.
Tools that load a model call check() in-process. Scoring is exempt: the
streaming referee streams by design.
"""
import argparse
import os
import pathlib
import sys

from vqlab.util.fs import GIB

HEADROOM = 0.90


def ram_bytes() -> int:
    """This box's physical RAM, read from the machine."""
    return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")


def bar_gib(headroom: float = HEADROOM) -> float:
    """The resident bar in GiB: the most a model may weigh to be loaded here."""
    return ram_bytes() * headroom / GIB


def check(artifact, headroom: float = HEADROOM) -> tuple[bool, int, int]:
    """(fits, artifact bytes, RAM bytes) for a resident load of `artifact`."""
    P = pathlib.Path(artifact)
    if not P.is_dir():
        raise SystemExit(f"PREFLIGHT FAIL: {P} is not a directory")
    art = sum(f.stat().st_size for f in P.glob("*.safetensors"))
    ram = ram_bytes()
    return art <= ram * headroom, art, ram


def report(artifact, headroom: float = HEADROOM) -> tuple[bool, str]:
    """check(), with the verdict as the text main() prints."""
    ok, art, ram = check(artifact, headroom)
    text = (f"PREFLIGHT  artifact {art / GIB:.3f} GiB   box RAM {ram / GIB:.0f} GiB   "
            f"bar {ram * headroom / GIB:.1f} GiB ({headroom:.0%})   -> {'OK' if ok else 'FAIL'}")
    if not ok:
        text += (f"\nPREFLIGHT FAIL: {pathlib.Path(artifact).name} needs {art / GIB:.3f} GiB "
                 f"resident and this box has {ram / GIB:.0f} GiB. A generation smoke or "
                 "speed test CANNOT run here — it will thrash swap and produce no verdict."
                 "\n  Run it on a box with more RAM. Scoring is exempt: the streaming "
                 "referee streams by design.")
    return ok, text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact")
    ap.add_argument("--headroom", type=float, default=HEADROOM,
                    help="fraction of physical RAM the artifact may occupy. mlx's "
                         "own recommended max is ~0.875 of RAM; 0.90 is already "
                         "generous and a smoke needs the KV cache on top.")
    a = ap.parse_args(argv)
    ok, text = report(a.artifact, a.headroom)
    print(text)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
