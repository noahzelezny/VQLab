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


def test_trunk_row_is_as_wide_as_the_head_reads(monkeypatch, tmp_path):
    # 2026-10-17: Flash-Next's head reads hc streams per row (hc * D); the
    # smoke fed it D and died in norm_h
    import mlx.core as mx
    seen = []

    class Head:
        D, h_width = 3, 12

        def make_draft_cache(self):
            return None

        def _trunk(self, h, ids, cache):
            seen.append(h.shape)
            return mx.zeros((1,))

    spec = NS(name="fam", head="h", arch_module=lambda m: None,
              head_cls=lambda: NS(from_sidecar=lambda *a: Head()))
    monkeypatch.setattr(model_loader, "load_artifact",
                        lambda path, lazy=False: (NS(args=NS(vocab_size=7)), None))
    monkeypatch.setattr(mtp_smoke_head.registry, "resolve", lambda model, family: spec)
    side = tmp_path / "s.safetensors"
    side.write_bytes(b"x")
    assert mtp_smoke_head.main(["--model", "art", "--sidecar", str(side),
                                "--steps", "1", "--warmup", "0"]) == 0
    assert {s[-1] for s in seen} == {12}
