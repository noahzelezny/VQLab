"""Reading teacher profiles (families/<f>/teachers/<t>/profile.json, written
by `vqlab family-profile`), in one place for every layer: the fit store
here in core, and family/registry.py, which adds the checkout-then-user
precedence on top.
"""
from __future__ import annotations

import json
import pathlib

from vqlab import config


def read_profile(p) -> dict | None:
    """A profile.json as a dict, or None when it is unreadable or not an object."""
    try:
        prof = json.loads(pathlib.Path(p).read_text())
    except (OSError, ValueError):
        return None
    return prof if isinstance(prof, dict) else None


def profiles_in(dirs) -> dict:
    """{(family, teacher): profile} for every readable profile under the
    family folders `dirs`, a later folder's overriding an earlier one's. An
    unreadable profile is skipped, not fatal; a missing folder is empty."""
    out = {}
    for d in map(pathlib.Path, dirs):
        for p in sorted(d.glob("*/teachers/*/profile.json")) if d.is_dir() else ():
            prof = read_profile(p)
            if prof is not None:
                out[(p.parent.parent.parent.name, p.parent.name)] = prof
    return out


def download_path(repo: str) -> str:
    """Where a Hub download of `repo` lands, in the recorded form every box
    expands (<teachers>/<org>--<name>)."""
    return "<teachers>/" + repo.replace("/", "--", 1)


def teacher_source(identity: dict) -> str | None:
    """The teacher's weights on this machine, in recorded form: the profile's
    own path, else the folder a download of its hf_repo lands in; None when
    neither is here. The one answer the picker and the compiler both use."""
    repo = identity.get("hf_repo")
    for p in (identity.get("path"), download_path(repo) if repo else None):
        local = config.expand(p) if p else ""
        if local and not local.startswith("hf://") and pathlib.Path(local).is_dir():
            return p
    return None
