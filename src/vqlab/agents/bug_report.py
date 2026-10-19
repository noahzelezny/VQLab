"""`vqlab lab report-bug`: a GitHub issue, filled in with what a bug needs.

Builds the issue body from the versions that decide behaviour (vqlab,
Python, macOS, the chip and its memory, mlx, mlx-lm, knurlogic,
huggingface_hub) and the last command that failed, then prints the
new-issue link (and, with --open, opens it). Nothing is sent: the browser
shows the issue for the user to read, edit and submit. Every path in it is
written in its portable form (config.portable_text), and argv is left out:
it names the user's files.
"""
from __future__ import annotations

import argparse
import platform
import sys
import urllib.parse

from vqlab import config

PACKAGES = ("mlx", "mlx-lm", "knurlogic", "huggingface_hub")
MAX_URL = 7000                  # GitHub refuses longer new-issue links


def issues_url() -> str:
    """The project's issue tracker, from its metadata (pyproject
    [project.urls] Issues): the installed package's, else the checkout's."""
    from importlib.metadata import PackageNotFoundError, metadata
    try:
        for u in metadata("vqlab").get_all("Project-URL") or []:
            name, _, url = u.partition(", ")
            if name == "Issues":
                return url
    except PackageNotFoundError:
        pass
    import tomllib

    from vqlab import _layout
    pp = _layout.SRC.parent / "pyproject.toml"
    return tomllib.loads(pp.read_text())["project"]["urls"]["Issues"]


def environment() -> list[str]:
    """`name: version` lines for what decides behaviour; no paths."""
    import vqlab
    from vqlab.agents import doctor
    lines = [f"vqlab: {vqlab.__version__}", f"python: {platform.python_version()}",
             f"macOS: {platform.mac_ver()[0] or platform.platform()}"]
    m = doctor.metal_info()
    lines.append(f"chip: {m.get('device')}, {m.get('memory_gib')} GiB" if m.get("available")
                 else "chip: Metal unavailable")
    lines += [f"{p}: {doctor.package_version(p) or 'not installed'}" for p in PACKAGES]
    return lines


def last_failure() -> dict | None:
    """The most recent logged command that ended with an error: its name,
    exit status and error text (portable), never its arguments."""
    from vqlab.records import runlog
    for r in reversed(runlog.runs()):
        rc = (r.get("end") or {}).get("rc")
        if rc not in (None, 0):
            return {"cmd": r.get("cmd"), "time": r.get("time"),
                    "rc": config.portable_text(str(rc))}
    return None


def body(what: str = "") -> str:
    """The issue text: the user's description, then the facts."""
    out = ["**What happened, and what did you expect?**", "", what or "<describe it here>", "",
           "**Command and output** (paste them; check for anything private first)", "",
           "```", "", "```", "", "**Environment**", "", "```", *environment(), "```"]
    fail = last_failure()
    if fail:
        out += ["", "**Last failed command**", "", "```",
                f"vqlab {fail['cmd']}  ({fail['time']})", f"exit: {fail['rc']}", "```"]
    return "\n".join(out) + "\n"


def new_issue_url(title: str, text: str) -> str:
    """GitHub's prefilled new-issue link; the body is cut to fit the limit."""
    base = issues_url().rstrip("/") + "/new?"
    q = {"title": title, "labels": "bug", "body": text}
    url = base + urllib.parse.urlencode(q)
    while len(url) > MAX_URL and len(q["body"]) > 200:
        q["body"] = q["body"][: len(q["body"]) * 3 // 4] + "\n…(cut to fit the link)\n"
        url = base + urllib.parse.urlencode(q)
    return url


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vqlab lab report-bug", description=__doc__.split("\n")[0])
    ap.add_argument("--title", default="", help="one line saying what went wrong")
    ap.add_argument("--what", default="", help="what happened and what you expected")
    ap.add_argument("--open", action="store_true", help="open the issue in the browser")
    a = ap.parse_args(argv)
    text = body(a.what)
    url = new_issue_url(a.title or "bug: ", text)
    print(text)
    print(f"Open this to file it on GitHub (nothing is sent until you submit):\n{url}")
    if a.open:
        import webbrowser
        webbrowser.open(url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
