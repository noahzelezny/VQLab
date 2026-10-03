"""mtp-arms builds one symlink arm per head and never touches the trunk."""
import os

from vqlab.bench import mtp_arms


def test_arms(tmp_path):
    t = tmp_path / "Trunk"
    t.mkdir()
    for f in ("config.json", "model.py", "model-1.safetensors", "mtp-head-q6.safetensors"):
        (t / f).write_text(f)
    h = tmp_path / "other-head.safetensors"
    h.write_text("h")
    before = sorted(os.listdir(t))
    mtp_arms.main(["--artifact", str(t), "--head", f"x={h}", "--head", f"q6={t / 'mtp-head-q6.safetensors'}",
                   "--models-dir", str(tmp_path / "models")])
    assert sorted(os.listdir(t)) == before
    arm = tmp_path / "models" / "ab-x--Trunk"
    heads = [f for f in os.listdir(arm) if f.startswith("mtp-head")]
    assert heads == ["mtp-head.safetensors"] and os.path.realpath(arm / heads[0]) == str(h.resolve())
    assert (arm / "model-1.safetensors").is_symlink()
