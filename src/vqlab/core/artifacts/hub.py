"""Hugging Face Hub comparisons: what a local artifact holds vs its repo.

Two callers, two reports, one fetch and one hashing rule:
  * `hub_diff` (here; `vqlab release-prep`): which local files are new or
    changed and which exist only on the Hub, recursive.
  * records/registry.hub_diff (`vqlab list --hub`): a per-file status table
    for the registry; it shares `remote_files` and `git_blob_sha1`.
Both need HF auth for private repos (HF_HOME / HF_TOKEN as for `hf`).
"""
from __future__ import annotations

from vqlab.util.hashing import git_blob_sha1, sha256_file

LFS_MIN = 10 << 20          # files above this are LFS on the Hub


def remote_files(repo: str):
    """-> (ModelInfo, {rfilename: sibling}) with file metadata (blob ids,
    sizes, LFS sha256). Raises huggingface_hub's RepositoryNotFoundError."""
    from huggingface_hub import HfApi
    info = HfApi().model_info(repo, files_metadata=True)
    return info, {s.rfilename: s for s in info.siblings}


def hub_diff(art, repo, deep=False):
    """-> (changed_or_new, only_on_hub, repo_exists). Needs HF auth for
    private repos (HF_HOME / HF_TOKEN as for `hf`)."""
    from huggingface_hub.utils import RepositoryNotFoundError
    try:
        _, hub = remote_files(repo)
    except RepositoryNotFoundError:
        return None, None, False
    local = [p for p in sorted(art.rglob("*")) if p.is_file()
             and p.relative_to(art).as_posix() not in ()
             and not any(part == "__pycache__" or part.startswith(".") for part in p.relative_to(art).parts)
             and not p.name.endswith(".pyc")]
    changed = []
    for p in local:
        rel = p.relative_to(art).as_posix()
        h = hub.get(rel)
        if h is None:
            changed.append(rel)
            continue
        size = p.stat().st_size
        if h.size != size:
            changed.append(rel)
        elif size < LFS_MIN or h.lfs is None:
            if h.blob_id and git_blob_sha1(p) != h.blob_id:
                changed.append(rel)
        elif deep and h.lfs.sha256 != sha256_file(p):
            changed.append(rel)
    have = {p.relative_to(art).as_posix() for p in local}
    only_hub = sorted(k for k in hub if k not in have and k != ".gitattributes")
    return changed, only_hub, True
