"""The package version is stated twice (pyproject.toml and vqlab.__version__);
they must agree, or the wheel and the build records disagree about what ran."""
import pathlib
import tomllib

import vqlab


def test_versions_agree():
    pp = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"
    assert tomllib.loads(pp.read_text())["project"]["version"] == vqlab.__version__
