"""runtime/ is shipped verbatim inside model bundles: no refactor may change
its bytes. Every file under src/vqlab/runtime/ is hashed against the snapshot
in tests/fixtures/runtime_sha256.json, and every earlier revision listed in
runtime/equivalent_revisions.json must still carry the bytes it names.

A deliberate runtime change (a kernel edit with its own gates) updates the
snapshot in the same commit:

    PYTHONPATH=src python tests/test_runtime_frozen.py --write
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "src" / "vqlab" / "runtime"
SNAPSHOT = ROOT / "tests" / "fixtures" / "runtime_sha256.json"


def _hashes():
    return {p.relative_to(RUNTIME).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(RUNTIME.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts}


def test_runtime_bytes_unchanged():
    want = json.loads(SNAPSHOT.read_text())["sha256"]
    got = _hashes()
    changed = sorted(k for k in set(want) | set(got) if want.get(k) != got.get(k))
    assert not changed, f"shipped runtime bytes changed: {changed}"


def test_equivalent_revisions_carry_their_bytes():
    reg = json.loads((RUNTIME / "equivalent_revisions.json").read_text())
    for rev in reg["revisions"]:
        text = (RUNTIME / rev["file"]).read_bytes()
        assert hashlib.sha256(text).hexdigest() == rev["sha256"], rev["file"]


if __name__ == "__main__" and "--write" in sys.argv:
    SNAPSHOT.parent.mkdir(exist_ok=True)
    SNAPSHOT.write_text(json.dumps({"_doc": "sha256 of every shipped runtime file "
                                    "(tests/test_runtime_frozen.py).",
                                    "sha256": _hashes()}, indent=1) + "\n")
    print(f"wrote {SNAPSHOT}")
