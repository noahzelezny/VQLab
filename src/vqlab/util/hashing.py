"""Content hashes: files, byte strings, git blobs, and an input directory's
cheap identity (`fingerprint`)."""
from __future__ import annotations

import hashlib
import json
import os
import pathlib

# Bytes hashed from the head of a big file when a full hash is not worth the
# wall time (a 751 GB teacher). records.provenance uses the same value.
HEAD = 1 << 20
# The build record's filename inside an artifact (records.provenance.RECORD).
RECORD = "vqlab_provenance.json"


def sha256_file(path, limit: int | None = None, algo: str = "sha256",
                missing_ok: bool = False) -> str | None:
    """Hex digest of a file (its first `limit` bytes when given). With
    missing_ok, an unreadable or absent file gives None instead of raising."""
    try:
        h = hashlib.new(algo)
        with open(path, "rb") as f:
            if limit is not None:
                h.update(f.read(limit))
            else:
                for blk in iter(lambda: f.read(1 << 24), b""):
                    h.update(blk)
        return h.hexdigest()
    except OSError:
        if missing_ok:
            return None
        raise


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_sha1(path) -> str:
    """The id git (and the Hugging Face Hub) gives this file's bytes."""
    data = pathlib.Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def fingerprint(d, head: int = HEAD, record: str = RECORD) -> dict:
    """Identity of an INPUT directory, cheap enough for a 751 GB teacher:
    per-shard bytes + head hash, full sha of config/index/model.py, its own
    provenance id if it carries one, and the HF revision if it is a hub
    snapshot."""
    d = pathlib.Path(d)
    rp = pathlib.Path(os.path.realpath(d))
    out = {"path": str(d), "realpath": str(rp)}
    parts = rp.parts
    if "snapshots" in parts:                        # HF hub cache layout
        i = parts.index("snapshots")
        if i + 1 < len(parts):
            out["hf_revision"] = parts[i + 1]
            out["hf_repo"] = parts[i - 1].removeprefix("models--").replace("--", "/")
    if not rp.is_dir():
        return out
    rec = rp / record
    if rec.exists():
        out["provenance_id"] = json.loads(rec.read_text()).get("id")
    for name in ("config.json", "model.safetensors.index.json", "model.py"):
        if (rp / name).exists():
            out.setdefault("files", {})[name] = sha256_file(rp / name)
    shards = {}
    for f in sorted(rp.glob("*.safetensors")):
        r = pathlib.Path(os.path.realpath(f))
        shards[f.name] = {"bytes": r.stat().st_size, "head_sha256": sha256_file(r, head)}
    out["shards"] = shards
    return out
