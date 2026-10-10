"""read_doc: a slice of a lab document -- the routing layer (CONTEXT.md at
the root and in every stage folder), family data, the docs and the law book,
plus a private `lab/` notebook when the checkout has one. Never source code."""

from __future__ import annotations

from typing import Any, Dict

from vqlab.ops.errors import LabError
from vqlab.ops.roots import DOCS_URL, PKG, REPO

DOC_ALLOW = ("docs", "families", "CONTEXT.md", "AGENTS.md",
             "README.md", "METHODOLOGY.md", "REPRODUCING.md", "lab",
             "research/CONTEXT.md")


def read_doc(path: str, start: int = 1, lines: int = 200) -> Dict[str, Any]:
    """Read a slice of a lab doc (docs/, AGENTS.md, the law book, ...)."""
    rel = path.lstrip("/")
    # Stage contracts (src/vqlab/<stage>/CONTEXT.md) are docs; the source
    # beside them is not -- read_doc never serves code.
    is_contract = (rel.startswith("src/vqlab/") and rel.endswith("/CONTEXT.md")
                   and ".." not in rel)
    if not is_contract and not any(rel == a or rel.startswith(a.rstrip("/") + "/")
                                   for a in DOC_ALLOW):
        raise LabError("NOT_ALLOWED", f"{path} is not a readable lab doc",
                       allowed=list(DOC_ALLOW))
    # Stage contracts ship inside the package, so they resolve from PKG even
    # in a wheel install; everything else lives only in a source checkout.
    p = PKG / rel[len("src/vqlab/"):] if is_contract else REPO / rel
    if not p.is_file():
        if not is_contract and not (REPO / "docs").is_dir():
            url = f"{DOCS_URL}/{rel}"
            raise LabError("NO_CHECKOUT", f"{rel} is not installed: the docs ship only in a "
                           f"source checkout of VQLab, not in the wheel. Read it at {url}",
                           url=url)
        raise LabError("NOT_FOUND", f"{p} does not exist")
    all_lines = p.read_text(errors="replace").split("\n")
    start = max(1, start)
    chunk = all_lines[start - 1:start - 1 + max(1, min(lines, 1000))]
    return {"path": str(p), "start": start, "end": start + len(chunk) - 1,
            "total_lines": len(all_lines), "text": "\n".join(chunk)}
