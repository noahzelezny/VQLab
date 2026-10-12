"""Reading teacher profiles (families/<f>/teachers/<t>/profile.json, written
by `vqlab family-profile`), in one place for every layer: the fit store
here in core, and family/registry.py, which adds the checkout-then-user
precedence on top.
"""
from __future__ import annotations

import json
import pathlib


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
