"""tokenizer_match: a tokenizer.json with another model's pre_tokenizer fails."""
import json

import pytest

tokenizers = pytest.importorskip("tokenizers")

from vqlab.gate import tokenizer_match as tm


def _tok(pre):
    from tokenizers import Tokenizer, models, decoders
    t = Tokenizer(models.BPE(unk_token=None))
    t.pre_tokenizer = pre
    t.decoder = decoders.ByteLevel()
    return json.loads(t.to_str())


def test_match_and_mismatch(tmp_path):
    from tokenizers import pre_tokenizers
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps(_tok(pre_tokenizers.ByteLevel())))
    art = tmp_path / "art"; art.mkdir()
    (art / "README.md").write_text("---\nbase_model: org/base\n---\n")
    (art / "tokenizer.json").write_text(ref.read_text())
    assert tm.compare(art, ref) == []
    (art / "tokenizer.json").write_text(json.dumps(_tok(pre_tokenizers.Whitespace())))
    assert any("pre_tokenizer" in p for p in tm.compare(art, ref))
    assert tm.base_model(art) == "org/base"


def test_no_base_needs_ref(tmp_path):
    with pytest.raises(LookupError):
        tm.reference(tmp_path)
