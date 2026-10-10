"""What a numerics build stamp holds, and how two are compared and printed.

Pure functions over the stamp dict; no family knowledge. BUILDING a stamp
needs the family registry (runtime, scorer variant), so that lives in
family/version_stamp.py, which re-exports these names. Records and cards
that only read a stamp import from here.
"""
from __future__ import annotations

# The fields a cache and a run must agree on. Order is the report order.
COMPARED = ("mlx", "mlx_lm", "arch_sha256", "scorer_variant")


def diff(a: dict | None, b: dict | None) -> list:
    """[(field, a, b)] for every COMPARED field that differs."""
    a, b = a or {}, b or {}
    return [(k, a.get(k), b.get(k)) for k in COMPARED if a.get(k) != b.get(k)]


def describe(st: dict | None) -> str:
    """'mlx X, mlx-lm Y, variant Z' -- the card line."""
    if not st:
        return "build not stamped"
    return (f"mlx {st.get('mlx')}, mlx-lm {st.get('mlx_lm')}, "
            + (f"knurlogic {st['knurlogic']}, " if st.get("knurlogic") else "")
            + f"variant {st.get('scorer_variant') or 'none'}")
