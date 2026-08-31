"""VQLab: size-targeted vector-quantized builds of large models on Apple
Silicon (MLX), and MTP speculative decoding over stock mlx-lm.

    from mlx_lm import load
    from vqlab import load_mtp_head, mtp_generate

    model, tok = load(path, trust_remote_code=True)
    head, _ = load_mtp_head(model, model_path=path)
    print(mtp_generate(model, tok, "Explain VQ.", head, temp=0.7))

See README.md and METHODOLOGY.md.
"""

__version__ = "0.1.0"

from .mtp import (  # noqa: E402
    FAMILIES,
    FamilySpec,
    MTPResponse,
    load_mtp_head,
    mtp_generate,
    mtp_stream_generate,
    register,
)

__all__ = ["FAMILIES", "FamilySpec", "MTPResponse", "load_mtp_head",
           "mtp_generate", "mtp_stream_generate", "register", "__version__"]
