"""VQLab: size-targeted vector-quantized builds of large models on Apple
Silicon (MLX), and MTP speculative decoding over stock mlx-lm.

    from mlx_lm import load
    from vqlab import load_mtp_head, mtp_generate

    model, tok = load(path, trust_remote_code=True)
    head, _ = load_mtp_head(model, model_path=path)
    print(mtp_generate(model, tok, "Explain VQ.", head, temp=0.7))

See README.md and METHODOLOGY.md.
"""

__version__ = "0.2.0"

from . import _layout  # noqa: E402,F401  stage folders + pre-split import names
from .family import arch as _arch  # noqa: E402

# deepseek_v4 / qwen4_exp / Knurlogic's qwen3_5, gemma4, glm5_next as
# mlx_lm.models.<name> (family/arch/__init__.py; VQLAB_VENDORED_ARCH=0 for stock)
_arch.install()

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
