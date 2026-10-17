"""core/source/fit_store.py: which profiles the store reads to name a
teacher and attribute a fit."""
import json

from vqlab import config
from vqlab.core.source import fit_store as fs


def _profile(root, family, teacher, **body):
    d = root / family / "teachers" / teacher
    d.mkdir(parents=True)
    (d / "profile.json").write_text(json.dumps({"family": family, "teacher": teacher, **body}))


def test_checked_in_profiles_attribute_a_fit(tmp_path, monkeypatch):
    # 2026-10-17: the store read only the user's (empty) folder, so filing
    # 768 fits left every one unattributed while the profiles sat in families/
    checkout, user = tmp_path / "checkout", tmp_path / "user"
    _profile(checkout, "qwen3_5", "Org--T", modules={"signatures": {"down_proj": [[2, 128, 64]]}})
    monkeypatch.setattr(config, "checkout_families_dir", lambda: checkout)
    monkeypatch.setattr(config, "families_dir", lambda: user)
    sig = fs.load_signatures()
    rec = {"proj": "down_proj", "E": 2, "OUT": 128, "IN": 64}
    assert fs.attribute(rec, sig) == ("qwen3_5", "Org--T")
    # an explicit folder is read alone
    assert fs.attribute(rec, fs.load_signatures(user)) == ("_unattributed", "no-matching-profile")


def test_a_teacher_copy_is_named_by_a_checked_in_profile(tmp_path, monkeypatch):
    checkout, user = tmp_path / "checkout", tmp_path / "user"
    t = tmp_path / "some-copy"
    t.mkdir()
    (t / "config.json").write_text("{}")
    (t / "model.safetensors").write_bytes(b"\0" * 64)
    csha, shard_fp = fs.teacher_identity(t)
    _profile(checkout, "qwen3_5", "Org--T",
             identity={"config_sha256": csha, "shard_fingerprint": shard_fp})
    monkeypatch.setattr(config, "checkout_families_dir", lambda: checkout)
    monkeypatch.setattr(config, "families_dir", lambda: user)
    fs._SLUGS.clear()
    assert fs.teacher_slug(t) == "Org--T"


def test_a_table_fit_is_described_and_counted(tmp_path):
    # the n-gram (PLE) tables are 2-D [rows, ...], not [E, OUT, ...]; filing
    # 333 GiB of them died on an IndexError (2026-10-17)
    np = __import__("numpy")
    from safetensors.numpy import save_file

    from vqlab.core.artifacts.artifact import read_header
    m = "model.language_model.layers.1.ple.ngram_embedding.shard_0.weight"
    p = tmp_path / "ple.safetensors"
    codes = np.arange(6 * 10, dtype=np.uint16).reshape(6, 10) % 16
    save_file({m + ".codebook": np.ones((16, 4), np.float16), m + ".codes": codes,
               m + ".vq_scales": np.ones((6, 2), np.float16)}, str(p))
    h, base = read_header(p, with_base=True)
    d = fs.describe(p, h, base, m)
    assert (d["E"], d["OUT"], d["IN"], d["layer"], d["proj"]) == (1, 6, 40, 1, "shard_0")
    use = fs.code_usage({**d, "location": str(p)})
    assert use["lookups"] == 60 and use["dead"] == 0
