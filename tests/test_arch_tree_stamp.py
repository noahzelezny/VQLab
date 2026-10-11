"""arch_tree_sha256: the numerics stamp covers the arch file AND the arch
modules it imports (a MoE subclass file imports its base file, which holds the
attention / GatedDeltaNet code). Synthetic arch packages under a fake runtime
name; CPU only, no mlx."""
import json
import sys

import pytest

from vqlab.family import version_stamp as N

RT = "fakert_tree"


@pytest.fixture
def pkg(tmp_path, monkeypatch):
    root = tmp_path / RT
    models = root / "models"
    models.mkdir(parents=True)
    (root / "__init__.py").write_text("")
    (models / "__init__.py").write_text("")
    (models / "base.py").write_text("class BaseModelArgs: pass\n")
    (models / "mybase.py").write_text(
        "import math\nfrom .base import BaseModelArgs\n\ndef attn(x):\n    return x\n")
    (models / "mymoe.py").write_text(
        "from dataclasses import dataclass\nfrom .mybase import attn\n")
    (models / "unrelated.py").write_text("X = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    for m in [m for m in sys.modules if m == RT or m.startswith(RT + ".")]:
        monkeypatch.delitem(sys.modules, m)
    return models


def test_subclass_tree_covers_its_base_transitively(pkg):
    files = N.arch_files("mymoe", RT)
    assert set(files) == {"mymoe", "mybase", "base"}
    assert files["mybase"] == N._sha(pkg / "mybase.py")
    assert N.tree_sha256(files) and len(N.tree_sha256(files)) == 64


def test_changing_the_base_changes_the_tree_not_the_file_hash(pkg):
    st = N.arch_files("mymoe", RT)
    t0, f0 = N.tree_sha256(st), N._sha(pkg / "mymoe.py")
    (pkg / "mybase.py").write_text("def attn(x):\n    return x * 1\n")
    st2 = N.arch_files("mymoe", RT)
    assert N._sha(pkg / "mymoe.py") == f0                  # arch_sha256 blind
    assert N.tree_sha256(st2) != t0                        # tree sees it
    assert "base" not in st2                               # deps follow the new file


def test_check_cache_compares_the_tree(pkg):
    run = {"mlx": "1", "mlx_lm": "1", "model_type": "mymoe", "scorer_variant": None,
           "arch_sha256": "a" * 64, "arch_tree_sha256": "b" * 64}
    assert N.check_cache({"numerics": dict(run)}, run)["status"] == "match"
    with pytest.raises(SystemExit, match="arch_tree_sha256"):
        N.check_cache({"numerics": {**run, "arch_tree_sha256": "c" * 64}}, run)


def test_old_stamp_without_tree_is_accepted_with_note(capsys):
    run = {"mlx": "1", "mlx_lm": "1", "model_type": "mymoe", "scorer_variant": None,
           "arch_sha256": "a" * 64, "arch_files": {}, "arch_tree_sha256": "b" * 64}
    old = {k: v for k, v in run.items() if k not in ("arch_files", "arch_tree_sha256")}
    assert N.check_cache({"numerics": old}, run)["status"] == "match"
    assert "UNKNOWN" in capsys.readouterr().out
    # still compares what it has
    with pytest.raises(SystemExit, match="arch_sha256"):
        N.check_cache({"numerics": {**old, "arch_sha256": "d" * 64}}, run)


def test_tree_equivalence_entry_is_field_scoped(tmp_path):
    reg = tmp_path / "eq.json"
    reg.write_text(json.dumps({"entries": [{
        "model_type": "mymoe", "field": "arch_tree_sha256",
        "old_sha256": "1" * 64, "new_sha256": "2" * 64, "evidence": "x", "date": "d"}]}))
    a = {"model_type": "mymoe", "arch_tree_sha256": "1" * 64}
    b = {"model_type": "mymoe", "arch_tree_sha256": "2" * 64}
    assert N.arch_equivalence(a, b, [("arch_tree_sha256", "1" * 64, "2" * 64)], reg)
    # the same pair as a FILE hash is not excused by a tree entry
    assert N.arch_equivalence(a, b, [("arch_sha256", "1" * 64, "2" * 64)], reg) is None


def test_shipped_qwen3_5_moe_tree_entry_accepts():
    e = [e for e in json.loads(N.ARCH_EQUIVALENCE.read_text())["entries"]
         if e.get("field") == "arch_tree_sha256"][0]
    run = {"mlx": "1", "mlx_lm": "1", "model_type": e["model_type"], "scorer_variant": None,
           "arch_sha256": "f" * 64, "arch_tree_sha256": e["new_sha256"]}
    rec = N.check_cache({"numerics": {**run, "arch_tree_sha256": e["old_sha256"]}}, run)
    assert rec["status"] == "arch-equivalent"
