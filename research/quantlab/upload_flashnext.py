#!/usr/bin/env python3
"""Publish the four Flash-Next VQ rungs into a HF collection.

Stages README/LICENSE/chart into each artifact dir, uploads, then creates
the collection and adds all four. Artifacts stay on disk (benchmarking
against 397B/GLM to follow).
"""
import os, shutil
os.environ.setdefault("HF_HOME", "/Volumes/Thunderbay SSD/Mlx_Models")
from huggingface_hub import HfApi

E = "/Volumes/Thunderbay SSD/Exo Models"
Q = os.path.expanduser("~/Documents/AgenicAI/quantlab")
LICENSE = ("/Volumes/Thunderbay SSD/Mlx_Models/hub/models--Qwen--Qwen3.8-"
           "Flash-Next/snapshots/de4b8e4d43b917e7706784d8bb445c9af86a3540/LICENSE")
CHART = os.path.join(Q, "research/flash-next/chart_ladder.png")
JOBS = [
    ("qwen4exp_vq_packed_mixL01", "Qwen3.8-Flash-Next-VQ-2.1bpw",
     "MODEL_CARD_FLASHNEXT_2_1bpw.md",
     "d8/K16384 + K256 PLE + hot-2 mix, 45.0 GiB: KL 390 vs q4's 294 at less than half the bytes"),
    ("qwen4exp_vq_packed_31mix6", "Qwen3.8-Flash-Next-VQ-3.2bpw",
     "MODEL_CARD_FLASHNEXT_3_2bpw.md",
     "d4/K2048 + hot-6 mix, 69.4 GiB: KL 123.5, prose below q5 at 60% of its size"),
    ("qwen4exp_vq_packed_92mix6", "Qwen3.8-Flash-Next-VQ-4.4bpw",
     "MODEL_CARD_FLASHNEXT_4_4bpw.md",
     "d2/K256 + hot-6 K1024 mix, 94.1 GiB: KL 50.3, beats q6 at 43 GiB less"),
    ("qwen4exp_vq_packed_d2k1024", "Qwen3.8-Flash-Next-VQ-5.5bpw",
     "MODEL_CARD_FLASHNEXT_5_5bpw.md",
     "d2/K1024, 111.6 GiB: KL 34.1, nears q8 at 66 GiB less"),
]
IGNORE = ["__pycache__/*", ".DS_Store", "*.tmp.safetensors", "*.pre_*"]

api = HfApi()
who = api.whoami()
assert (who.get("auth") or {}).get("accessToken", {}).get("role") in ("write", "admin")

uploaded = []
for src, repo, card, msg in JOBS:
    d = os.path.join(E, src)
    body = open(os.path.join(Q, card), encoding="utf-8").read()
    assert repo in body and "PENDING" not in body, f"card not ready: {card}"
    open(os.path.join(d, "README.md"), "w", encoding="utf-8").write(body)
    shutil.copy(LICENSE, os.path.join(d, "LICENSE"))
    shutil.copy(CHART, os.path.join(d, "chart_ladder.png"))
    rid = f"TheDrainFlorist/{repo}"
    api.create_repo(rid, repo_type="model", exist_ok=True)
    print(f"uploading {src} -> {rid} ...", flush=True)
    api.upload_folder(folder_path=d, repo_id=rid, repo_type="model",
                      ignore_patterns=IGNORE, commit_message=msg)
    print(f"COMMIT LANDED {repo}", flush=True)
    uploaded.append(rid)

col = api.create_collection(
    title="Qwen3.8-Flash-Next VQ (data-free, Apple Silicon)",
    description=("Four vector-quantized rungs of Qwen3.8-Flash-Next (335 GiB "
                 "bf16) for 64/96/128 GB Apple Silicon. Data-free k-means VQ "
                 "with leverage-guided per-layer mixing; KL-to-teacher led. "
                 "Built with VQLab."),
    exists_ok=True)
for rid in uploaded:
    api.add_collection_item(col.slug, item_id=rid, item_type="model",
                            exists_ok=True)
print("collection:", col.slug, flush=True)
print("ALL FOUR DONE", flush=True)
