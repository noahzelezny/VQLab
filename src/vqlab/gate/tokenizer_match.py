"""Release gate: the artifact's tokenizer must tokenize like the base model's.

2026-10-10: three Qwen families shipped a tokenizer.json whose pre_tokenizer
came from a different model's conversion (no \\p{M}), so Hindi took 19 tokens
instead of 11. The round-trip probe in check-release passed it: an English
sentence never touches the difference. This compares the pre_tokenizer,
normalizer and decoder against the base model's own tokenizer.json, then
encodes and decodes a multilingual sample through both.

The base model is the card's `base_model:` line; its tokenizer.json comes from
the Hub (cached by huggingface_hub), or from `--tokenizer-ref <file>`.
"""
from __future__ import annotations

import json
import pathlib
import re
import unicodedata

SAMPLE = [
    "Hello world, the café is open.",
    unicodedata.normalize("NFD", "café naïve résumé"),
    "नमस्ते दुनिया, आप कैसे हैं?",
    "สวัสดีครับ ยินดีต้อนรับ",
    "مَرْحَبًا بِالْعَالَم",
    "def f(x):\n    return x**2  # ok",
    "Tiếng Việt có dấu",
    "日本語のテキストと한국어 텍스트",
    "Здравствуй, мир! 123,456.78",
    "emoji 👩🏽‍💻 ok",
    "  multiple   spaces\n\n\ttabs",
    "ภาษาไทย ٱلْعَرَبِيَّة हिन्दी ગુજરાતી தமிழ் ລາວ ꦗꦮ",
]
PARTS = ("pre_tokenizer", "normalizer", "decoder")


def base_model(art: pathlib.Path) -> str | None:
    card = art / "README.md"
    if not card.exists():
        return None
    m = re.search(r"^base_model:\s*(\S+)", card.read_text(), re.M)
    return m.group(1) if m else None


def reference(art: pathlib.Path, ref: str | None = None) -> pathlib.Path:
    if ref:
        return pathlib.Path(ref)
    repo = base_model(art)
    if not repo:
        raise LookupError("card names no base_model: pass --tokenizer-ref")
    from huggingface_hub import hf_hub_download
    return pathlib.Path(hf_hub_download(repo, "tokenizer.json"))


def compare(art: pathlib.Path, ref_file: pathlib.Path) -> list[str]:
    """Problems found; empty means the artifact tokenizes like the base."""
    from tokenizers import Tokenizer
    mine_p = art / "tokenizer.json"
    if not mine_p.exists():
        return []                      # tokenizer-less artifacts are checked elsewhere
    mine, off = json.loads(mine_p.read_text()), json.loads(ref_file.read_text())
    probs = [f"{k} differs from the base model's" for k in PARTS
             if mine.get(k) != off.get(k)]
    a, b = Tokenizer.from_file(str(mine_p)), Tokenizer.from_file(str(ref_file))
    for s in SAMPLE:
        ia, ib = a.encode(s).ids, b.encode(s).ids
        if ia != ib:
            probs.append(f"encodes {s[:16]!r} to {len(ia)} tokens, base {len(ib)}")
        elif a.decode(ia) != b.decode(ib):
            probs.append(f"decodes {s[:16]!r} differently from the base")
    return probs


def check(art: pathlib.Path, ref: str | None = None) -> list[str]:
    return compare(art, reference(art, ref))
