#!/usr/bin/env python3
"""Publish the E91 flat-d4/K2048 flagship (2.3410 / 2.5963 @ 143.682 GiB).

Order is deliberate:
  1. Upload weights + card G in ONE commit to the EXISTING VQ-3.1bpw repo. The
     predecessor revision stays fetchable exactly as the card promises, and the
     repo never serves new weights under the old card's numbers.
  2. THEN rename to VQ-3bpw, leaving the old URL redirecting.
Uploading to a fresh repo instead would orphan the old URL AND break the card's
`--revision` pin, which points into THIS repo's history.

upload_folder (not upload_large_folder) is chosen for atomicity: it pre-uploads
every LFS blob and then makes a single commit, so downloaders never see a repo
half-way between two builds. Per logs_live_upload_101.SUMMARY.log, a retry
after the bytes are pre-uploaded commits in ~2 minutes.

That same log is why *.pre_* are ignored rather than moved: last time two
residue files were moved out AFTER the uploader registered them, and every
commit batch containing one failed - 15.4M failed-commit lines over 74 minutes.
"""
import os, sys
os.environ.setdefault("HF_HOME", "/Volumes/Thunderbay SSD/Mlx_Models")
from huggingface_hub import HfApi

SRC  = "/Volumes/Thunderbay SSD/Exo Models/rotlab--397B-flatk2048-refit-packed"
OLD  = "TheDrainFlorist/Qwen3.5-397B-A17B-VQ-3.1bpw"
NEW  = "TheDrainFlorist/Qwen3.5-397B-A17B-VQ-3bpw"
PRED = "a0da72a0c43932704a272fe3ce6a6513194570eb"
IGNORE = ["__pycache__/*", ".DS_Store", "*.pre_vision_config", "*.pre_total_size"]

api = HfApi()
w = api.whoami()
assert (w.get("auth") or {}).get("accessToken", {}).get("role") in ("write", "admin")
print("authenticated as", w["name"], flush=True)

# the card prints this pin; if it is not real the card lies
assert PRED in {c.commit_id for c in api.list_repo_commits(OLD, repo_type="model")}, \
    "predecessor revision missing from history"
print("predecessor pin verified in repo history", flush=True)

# the card and the bytes must agree about which artifact this is
card = open(os.path.join(SRC, "README.md"), encoding="utf-8").read()
assert "2.3410" in card and "__" not in card.replace("__PRED", ""), "card mismatch/placeholder"
print("staged card verified", flush=True)

if "--rename-only" not in sys.argv:
    print("uploading ~136 GiB of changed shards -> %s ..." % OLD, flush=True)
    api.upload_folder(
        folder_path=SRC, repo_id=OLD, repo_type="model", ignore_patterns=IGNORE,
        commit_message="E91 flat d4/K2048 refit: 2.3410 wikitext / 2.5963 code @ 143.682 GiB",
        commit_description=(
            "Same size and geometry as the build it replaces. Better on wikitext by "
            "0.0109 = 1.9x the measured fit-to-fit noise floor at this geometry, under "
            "our 3x bar, so it is reported and not claimed. Against the community "
            "3.5bit: 21.9 GiB smaller, better on prose (3.6x floor), tied on code. "
            "Predecessor weights remain fetchable at revision " + PRED + "."),
    )
    print("UPLOAD COMMIT LANDED", flush=True)

print("renaming %s -> %s" % (OLD, NEW), flush=True)
api.move_repo(from_id=OLD, to_id=NEW, repo_type="model")
print("RENAME DONE - old URL now redirects", flush=True)
