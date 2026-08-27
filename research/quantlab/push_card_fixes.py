#!/usr/bin/env python3
"""Push model-card corrections to the LIVE HF repos. README.md only, no weights.

    ./push_card_fixes.py --verify     # read-only: is every live card == local?
    ./push_card_fixes.py --dry-run    # show the diff that WOULD be pushed
    ./push_card_fixes.py              # push only the cards that differ, then verify

Default behaviour is deliberately conservative:
  * every repo id is checked against the live account listing before anything
    is sent, so a typo fails loudly instead of creating a stray repo;
  * COVERAGE is asserted -- if the account has a repo this map does not name,
    the run aborts. The old version of this file silently covered 7 of 13,
    which reads as "the set is synced" when it is not;
  * each card is diffed against its LIVE README and only pushed if it differs;
  * after pushing, every card is re-fetched and sha256-compared to local.

Why local filenames look nothing like repo names: the 397B cards predate the
bpw naming convention. The letters are LOCAL ONLY -- every card uploads as
README.md and carries its own H1. No letter is ever public. The mapping below
was verified by matching each live README's H1 against the local files.

HF_HOME (and the token) live on the SSD per ~/.zshrc; non-interactive shells
do not source it, so an unset HF_HOME pushes UNAUTHENTICATED.
"""
import sys, io, os, hashlib, difflib

os.environ.setdefault("HF_HOME", "/Volumes/Thunderbay SSD/Mlx_Models")
from huggingface_hub import HfApi, hf_hub_download

OWNER = "TheDrainFlorist"

# repo name -> local card. Verified 2026-08-25 by live-H1 match, all 13 live.
CARDS = {
    "Qwen3.5-397B-A17B-VQ-2.2bpw":  "MODEL_CARD_397B_F.md",
    "Qwen3.5-397B-A17B-VQ-2.4bpw":  "MODEL_CARD_397B_C.md",
    "Qwen3.5-397B-A17B-VQ-2.6bpw":  "MODEL_CARD_397B_K512.md",
    "Qwen3.5-397B-A17B-VQ-3.1bpw":  "MODEL_CARD_397B_G.md",
    "Qwen3.6-35B-A3B-VQ-3.4bpw":    "MODEL_CARD_QWEN_SMALL.md",
    "Qwen3.6-35B-A3B-VQ-3.8bpw":    "MODEL_CARD_QWEN36_3_8.md",
    "Qwen3.6-35B-A3B-VQ-4.6bpw":    "MODEL_CARD_QWEN_QUALITY.md",
    "Qwen3.6-35B-A3B-VQ-5.4bpw":    "MODEL_CARD_QWEN36_5_4.md",
    "Qwen3.8-27B-VQ-3.9bpw":        "MODEL_CARD_QWEN38_3_9bpw.md",
    "Qwen3.8-27B-VQ-4.5bpw":        "MODEL_CARD_QWEN38_4_5bpw.md",
    "Qwen3.8-27B-VQ-4.8bpw":        "MODEL_CARD_QWEN38_4_8bpw.md",
    "gemma-4-26b-a4b-it-VQ-6.2bpw": "MODEL_CARD_GEMMA_QUALITY.md",
    "gemma-4-e4b-it-VQ-PLE":        "MODEL_CARD_GEMMA_E4B_VQPLE.md",
}

# Local cards that are deliberately NOT pushed anywhere. Listed so that a
# future reader can tell "excluded on purpose" from "forgotten".
NOT_PUBLISHED = {
    "MODEL_CARD_397B_E.md":    "SUPERSEDED -- described VQ-3.1bpw, whose repo was "
                               "RENAMED twice (3.1->3->3.1, final 2026-08-26). HF REDIRECTS, so "
                               "pushing this by the old name silently overwrote the "
                               "flagship card once already. Never re-add it.",
    "MODEL_CARD_GEMMA_SMALL.md": "RETIRED 2026-08-20, never published; kept as ladder data.",
}

# Text that must never appear in a pushed card: retracted claims and stale
# values. Each entry is (needle, why). Cheap last line of defence.
FORBIDDEN = [
    ("5.19 and",   "retracted ppl claim: span is 3.6x the floor, not inside it"),
    ("0.859 GiB",  "stale tower size; measured value is 0.858"),
    ("3.1706",     "retired flat-K128 ladder point, unless labelled (v1 weights)"),
]

MSG = "card corrections (see paper/LEDGER.md for the audit that produced them)"


def body(path):
    return io.open(path, encoding="utf-8").read()


def live_readme(repo):
    p = hf_hub_download(f"{OWNER}/{repo}", "README.md", force_download=True)
    return io.open(p, encoding="utf-8").read()


EXCUSES = ("v1", "SUPERSEDED", "RETIRED", "predecessor", "before their repos")


def check_forbidden(path, text):
    """Flag retracted/stale text, but not when it is labelled as history.

    The label is often NOT on the same line as the value -- e.g. card F puts
    3.1706 in a v2-vs-v1 table whose "v1" label lives in the column header.
    So look at a window around the hit, not the line. (An earlier same-line
    version of this check produced exactly that false positive.)
    """
    hits, lines = [], text.splitlines()
    for needle, why in FORBIDDEN:
        for i, line in enumerate(lines):
            if needle not in line:
                continue
            window = "\n".join(lines[max(0, i - 6):i + 3])
            if any(e in window for e in EXCUSES):
                continue
            hits.append(f"{path}:{i+1}: contains {needle!r} -- {why}")
            break
    return hits


def main():
    verify_only = "--verify" in sys.argv
    dry = "--dry-run" in sys.argv
    api = HfApi()
    print("as:", api.whoami()["name"])

    live = {m.id.split("/")[1] for m in api.list_models(author=OWNER)}
    missing = live - set(CARDS)
    unknown = set(CARDS) - live
    if unknown:
        sys.exit(f"ABORT: map names repos that do not exist: {sorted(unknown)}")
    if missing:
        sys.exit("ABORT: these live repos are not in CARDS, so this run would "
                 f"NOT be full coverage: {sorted(missing)}\n"
                 "Add them (or document them) before pushing.")
    print(f"coverage OK: {len(CARDS)}/{len(live)} live repos mapped")

    problems = []
    for path in set(CARDS.values()):
        problems += check_forbidden(path, body(path))
    if problems:
        sys.exit("ABORT: forbidden text in cards:\n  " + "\n  ".join(problems))

    changed, clean = [], []
    for repo, path in sorted(CARDS.items()):
        loc, lv = body(path), live_readme(repo)
        if loc == lv:
            clean.append(repo); continue
        changed.append((repo, path))
        d = list(difflib.unified_diff(lv.splitlines(), loc.splitlines(),
                                      "LIVE", "LOCAL", lineterm="", n=0))
        print(f"\n=== {repo}  <- {path}  ({len(d)-2} diff lines)")
        for line in d[2:]:
            print("   ", line)

    print(f"\n{len(clean)} already in sync, {len(changed)} differ")
    if verify_only:
        print("VERIFY:", "all in sync" if not changed else "OUT OF SYNC (above)")
        return 0 if not changed else 1
    if dry or not changed:
        print("(dry run, nothing sent)" if dry else "nothing to push")
        return 0

    for repo, path in changed:
        api.upload_file(path_or_fileobj=body(path).encode(), path_in_repo="README.md",
                        repo_id=f"{OWNER}/{repo}", repo_type="model", commit_message=MSG)
        print("  pushed", path, "->", repo)

    print("\nverifying...")
    bad = 0
    for repo, path in changed:
        same = hashlib.sha256(live_readme(repo).encode()).hexdigest() == \
               hashlib.sha256(body(path).encode()).hexdigest()
        print(f"  {repo:32s} identical={same}")
        bad += not same
    print("ALL VERIFIED" if not bad else f"*** {bad} MISMATCH ***")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
