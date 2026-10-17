"""gate/mtp_smoke_head: the trunk loads through family.model_loader."""
from types import SimpleNamespace as NS

import pytest

from vqlab.family import model_loader
from vqlab.gate import mtp_smoke_head


class _Loaded(Exception):
    pass


def test_trunk_loads_lazily_through_load_artifact(monkeypatch):
    # 2026-10-15: a bare mlx_lm load() without trust_remote_code was refused
    # on every bundled VQ artifact, failing each MTP build queue at this step
    calls = []

    def fake(path, lazy=False):
        calls.append((path, lazy))
        raise _Loaded

    monkeypatch.setattr(model_loader, "load_artifact", fake)
    with pytest.raises(_Loaded):
        mtp_smoke_head.main(["--model", "art", "--sidecar", "s.safetensors"])
    assert calls == [("art", True)]


@pytest.mark.parametrize("model", [
    NS(args=NS(vocab_size=7)),                                    # mlx_lm text model
    NS(language_model=NS(args=NS(vocab_size=7))),                 # a wrapper's text model
    NS(args=NS(text_config={"vocab_size": 7})),                   # qwen4_exp: a dict
    NS(args=NS(vocab_size=None, text_config=NS(vocab_size=7))),   # an args object
])
def test_vocab_is_found_where_each_runtime_keeps_it(model):
    # 2026-10-16: the Flash-Next preflight's head smoke died on args.vocab_size
    assert mtp_smoke_head.vocab_size(model) == 7
