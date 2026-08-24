#!/usr/bin/env python3
"""Push the audited model-card corrections to the six LIVE HF repos.

Only README.md is touched; no weights. Each local file was synced FROM live
before editing (62a41f6), so the diff pushed is exactly the fix set in bd691a7.
Run `hf auth login` first. --dry-run to preview.
"""
import sys, io
from huggingface_hub import HfApi

CARDS = {
    "Qwen3.5-397B-A17B-VQ-2.2bpw":  "MODEL_CARD_397B_F.md",
    "Qwen3.5-397B-A17B-VQ-2.4bpw":  "MODEL_CARD_397B_C.md",
    "Qwen3.5-397B-A17B-VQ-3.1bpw":  "MODEL_CARD_397B_E.md",
    "Qwen3.6-35B-A3B-VQ-4.6bpw":    "MODEL_CARD_QWEN_QUALITY.md",
    "Qwen3.6-35B-A3B-VQ-3.4bpw":    "MODEL_CARD_QWEN_SMALL.md",
    "gemma-4-26b-a4b-it-VQ-6.2bpw": "MODEL_CARD_GEMMA_QUALITY.md",
}
MSG = ("card corrections: floor-check quality claims, fix repo name, "
       "retire superseded sweep row (audit 2026-08-24)")

dry = "--dry-run" in sys.argv
api = HfApi()
print("as:", api.whoami()["name"] if not dry else "(dry run)")
for repo, path in CARDS.items():
    rid = f"TheDrainFlorist/{repo}"
    body = io.open(path, encoding="utf-8").read()
    assert "__" not in body.replace("__PRED", ""), f"placeholder in {path}"
    print(f"  {'would push' if dry else 'pushing'} {path} -> {rid} ({len(body)} B)")
    if not dry:
        api.upload_file(path_or_fileobj=body.encode(), path_in_repo="README.md",
                        repo_id=rid, repo_type="model", commit_message=MSG)
print("done")
