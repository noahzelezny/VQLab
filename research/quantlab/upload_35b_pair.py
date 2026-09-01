#!/usr/bin/env python3
"""Publish the two 35B rungs: e94b (3.8bpw) and E140 (5.4bpw)."""
import os, io
os.environ.setdefault("HF_HOME", "/Volumes/Thunderbay SSD/Mlx_Models")
from huggingface_hub import HfApi
E = "/Volumes/Thunderbay SSD/Exo Models"
JOBS = [("e94b-35b-K8192-refit-0821-packed", "Qwen3.6-35B-A3B-VQ-3.8bpw",
         "d4/K8192, 15.670 GiB: KL 53.02 vs the community 4-bit's 78.56, at 3.3 GiB smaller"),
        ("e140-35b-d2K1024-packed", "Qwen3.6-35B-A3B-VQ-5.4bpw",
         "d2/K1024, 22.226 GiB: KL 28.14, a factor 1.4 below the interpolated affine frontier")]
IGNORE = ["__pycache__/*", ".DS_Store", "*.pre-e134", "*.pre-visionfix", "*.pre_*"]
api = HfApi()
assert (api.whoami().get("auth") or {}).get("accessToken", {}).get("role") in ("write", "admin")
for src, repo, msg in JOBS:
    rid = f"TheDrainFlorist/{repo}"
    card = io.open(os.path.join(E, src, "README.md"), encoding="utf-8").read()
    assert repo in card and "__" not in card, f"card wrong for {repo}"
    api.create_repo(rid, repo_type="model", exist_ok=True)
    print(f"uploading {src} -> {rid} ...", flush=True)
    api.upload_folder(folder_path=os.path.join(E, src), repo_id=rid, repo_type="model",
                      ignore_patterns=IGNORE, commit_message=msg)
    print(f"COMMIT LANDED {repo}", flush=True)
print("BOTH DONE", flush=True)
