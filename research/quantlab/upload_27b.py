#!/usr/bin/env python3
"""Publish the three Qwen3.8-27B rungs. No MLX quantization of this model
has been published by anyone, so these are the first."""
import os, io
os.environ.setdefault("HF_HOME", "/Volumes/Thunderbay SSD/Mlx_Models")
from huggingface_hub import HfApi
E = "/Volumes/Thunderbay SSD/Exo Models"
JOBS = [("e130-27b-d4K4096-vq-packed", "Qwen3.8-27B-VQ-3.9bpw",
         "d4/K4096, 12.47 GiB: KL 85.8 to bf16, less than half the 3-bit conversion's 187.8"),
        ("e124-27b-dense-d2K256-vq-packed", "Qwen3.8-27B-VQ-4.5bpw",
         "d2/K256, 14.45 GiB: KL 40.3 vs the 4-bit conversion's 45.8, at 0.50 GiB smaller"),
        ("e142-27b-d2K512-iters30-vq-packed", "Qwen3.8-27B-VQ-4.8bpw",
         "d2/K512, 15.45 GiB: KL 32.8, 28% closer to bf16 than the 4-bit at +3.3% bytes; seeded fit")]
IGNORE = ["__pycache__/*", ".DS_Store", "*.pre_total_size", "*.pre-visionfix", "*.pre_*"]
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
print("ALL THREE DONE", flush=True)
