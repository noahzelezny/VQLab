#!/usr/bin/env python3
"""Publish flat d4/K512 as Qwen3.5-397B-A17B-VQ-2.6bpw.

New repo, so upload_folder makes one atomic commit into an empty repo -
no window where the repo serves a half-built model.

Residue is EXCLUDED, never moved: moving *.pre_* out mid-upload after the
uploader registered them cost 74 minutes and 15.4M failed-commit lines on
the 101 GiB push (logs_live_upload_101.SUMMARY.log).
"""
import os, sys
os.environ.setdefault("HF_HOME", "/Volumes/Thunderbay SSD/Mlx_Models")
from huggingface_hub import HfApi

SRC  = "/Volumes/Thunderbay SSD/Exo Models/rotlab--397B-flatk512-packed"
REPO = "TheDrainFlorist/Qwen3.5-397B-A17B-VQ-2.6bpw"
IGNORE = ["__pycache__/*", ".DS_Store", "*.pre_total_size", "*.pre_vision_config"]

api = HfApi()
w = api.whoami()
assert (w.get("auth") or {}).get("accessToken", {}).get("role") in ("write", "admin")
print("authenticated as", w["name"], flush=True)

card = open(os.path.join(SRC, "README.md"), encoding="utf-8").read()
assert "2.5634" in card and "VQ-2.6bpw" in card and "__" not in card, "card wrong/placeholder"
print("staged card verified", flush=True)

api.create_repo(REPO, repo_type="model", exist_ok=True)
print("repo ready:", REPO, flush=True)

api.upload_folder(
    folder_path=SRC, repo_id=REPO, repo_type="model", ignore_patterns=IGNORE,
    commit_message="flat d4/K512: 2.5634 wikitext / 2.6123 code @ 122.3 GiB",
    commit_description=(
        "Beats the community 2.6-bit build on both corpora at comparable size: "
        "19.5% better prose (24x the fit-to-fit floor), 2.0% better code (3.1x). "
        "Floor is borrowed from a K256 geometry - none measured at K512. "
        "Includes the bf16 vision tower (0.849 GiB) that the comparator omits, so "
        "the like-for-like size delta is +0.88 GiB rather than +1.7. Serves on a "
        "2-node exo ring; does not fit a single 128 GB machine."),
)
print("UPLOAD COMMIT LANDED", flush=True)
