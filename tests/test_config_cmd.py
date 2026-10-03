"""vqlab config init writes a config that the resolver reads back, refuses to
overwrite, and creates the storage directories."""
import pytest

from vqlab import config as C
from vqlab.agents import config_cmd


def test_init_roundtrip(tmp_path, monkeypatch):
    for v in C._ENV.values():
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("VQLAB_CONFIG", str(tmp_path / "c.toml"))
    C._file_paths.cache_clear()
    config_cmd.main(["init", "--root", str(tmp_path / "lab")])
    C._file_paths.cache_clear()
    assert C.scratch() == tmp_path / "lab" / "scratch" and C.scratch().is_dir()
    assert C.fit_store() == [tmp_path / "lab" / "fits"]
    with pytest.raises(SystemExit):
        config_cmd.main(["init", "--root", str(tmp_path / "other")])
    C._file_paths.cache_clear()
