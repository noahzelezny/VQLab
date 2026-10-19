"""fit/fit_dense_vq: the dense fitter's command line."""
import pytest

from vqlab.family import registry
from vqlab.fit import fit_dense_vq


def test_offers_every_registered_dense_family(monkeypatch, capsys):
    # a dense family is DATA (registry data entries); fit-dense hardcoded two
    monkeypatch.setitem(registry.DENSE_FAMILIES, "data_fam", ("m.{li}.{key}.weight", "0-0"))
    with pytest.raises(SystemExit):
        fit_dense_vq.main(["--help"])
    assert "data_fam" in capsys.readouterr().out
