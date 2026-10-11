"""What a numerics build stamp holds, and how two are compared and printed.

Pure functions over the stamp dict; no family knowledge. BUILDING a stamp
needs the family registry (runtime, scorer variant), so that lives in
family/version_stamp.py, which re-exports these names. Records and cards
that only read a stamp import from here.
"""
from __future__ import annotations

# The fields a cache and a run must agree on. Order is the report order.
COMPARED = ("mlx", "mlx_lm", "arch_sha256", "scorer_variant")


# The arch hash over the arch file AND the arch modules it imports. When both
# stamps carry it, it is compared in arch_sha256's place (it covers that file);
# a stamp written before it existed is compared on arch_sha256 alone.
TREE = "arch_tree_sha256"


def tree_compared(a: dict | None, b: dict | None) -> bool:
    """True when both stamps carry arch_tree_sha256 (so it is what is compared)."""
    return TREE in (a or {}) and TREE in (b or {})


def diff(a: dict | None, b: dict | None) -> list:
    """[(field, a, b)] for every COMPARED field that differs (arch_sha256 is
    replaced by arch_tree_sha256 when both stamps carry it)."""
    tree = tree_compared(a, b)
    a, b = a or {}, b or {}
    keys = [TREE if (k == "arch_sha256" and tree) else k for k in COMPARED]
    return [(k, a.get(k), b.get(k)) for k in keys if a.get(k) != b.get(k)]


def describe(st: dict | None) -> str:
    """'mlx X, mlx-lm Y, variant Z' -- the card line."""
    if not st:
        return "build not stamped"
    return (f"mlx {st.get('mlx')}, mlx-lm {st.get('mlx_lm')}, "
            + (f"knurlogic {st['knurlogic']}, " if st.get("knurlogic") else "")
            + f"variant {st.get('scorer_variant') or 'none'}")
