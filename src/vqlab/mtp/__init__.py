"""MTP speculative decoding as a decode strategy over stock mlx-lm.

    from vqlab import load_mtp_head, mtp_generate
    from mlx_lm import load

    model, tok = load(path, trust_remote_code=True)
    head, _ = load_mtp_head(model, model_path=path)
    print(mtp_generate(model, tok, "Explain VQ.", head, temp=0.7))

Adding a model family is a `FamilySpec` in registry.py plus a head module;
see that docstring. `vqlab mtp-pack` builds a sidecar, `vqlab mtp-bench`
measures the speedup, the acceptance and the near-tie divergence gate.
"""
from .loop import MTPResponse, load_mtp_head, mtp_generate, mtp_stream_generate
from .registry import FAMILIES, FamilySpec, register, resolve
from .sampling import Distribution, make_distribution, rejection_correct

__all__ = [
    "MTPResponse", "load_mtp_head", "mtp_generate", "mtp_stream_generate",
    "FAMILIES", "FamilySpec", "register", "resolve",
    "Distribution", "make_distribution", "rejection_correct",
]
