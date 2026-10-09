"""The public repo carries no machine paths, volume names, hostnames or home
directories: storage comes from vqlab.config. Noah has asked for this more
than once; this test is what keeps it true."""
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
# Assembled from fragments so a history rewrite that scrubs these strings
# (git filter-repo --replace-text) cannot rewrite the guard itself.
_TERMS = ["Thunder" + "bay", "Noahs" + "MacStudio", "Noahs" + "-Mac", "Nozzle" + "Book",
          "Noah's" + " Mac", "/Users/" + "noahzelezny", "/opt/" + "anaconda3"]
BANNED = re.compile("|".join(re.escape(t) for t in _TERMS))
# Patterns (also fragment-built): private LAN IPs, the owner's GitHub-less
# username as a home dir, lab host literals, hardcoded homebrew tool paths.
_PATTERNS = [
    r"\b10\.0\.0\.\d+\b", r"\b192\.168\.\d+\.\d+\b",
    r"\b172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+\b",
    r"/Users/(?!dev/|you/|user/|name/)[a-z][\w.-]+/",
    "noahs" + r"-m\d", "Mac" + r" ?Studio-\d", r"\b(?:m3|m4)\.local\b",
    r"/opt/homebrew/bin/",
]
PATTERNS = re.compile("|".join(_PATTERNS))


def test_no_machine_paths_in_tracked_files():
    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                           text=True, check=True).stdout.split()
    hits = []
    for f in files:
        if f == "tests/test_no_machine_paths.py":
            continue
        try:
            text = (ROOT / f).read_text(errors="ignore")
        except (IsADirectoryError, FileNotFoundError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if BANNED.search(line) or PATTERNS.search(line):
                hits.append(f"{f}:{n}: {line.strip()[:100]}")
    assert not hits, "machine paths in tracked files:\n" + "\n".join(hits)
