"""VQLab: size-targeted vector-quantized builds of large models on Apple
Silicon (MLX), and MTP speculative decoding over stock mlx-lm.

    import vqlab
    vqlab.use_knurlogic_arch()      # deepseek_v4 / qwen4_exp / Knurlogic's edits
    from mlx_lm import load
    from vqlab import load_mtp_head, mtp_generate

    model, tok = load(path, trust_remote_code=True)
    head, _ = load_mtp_head(model, model_path=path)
    print(mtp_generate(model, tok, "Explain VQ.", head, temp=0.7))

`import vqlab` changes nothing outside vqlab. `use_knurlogic_arch()` is the
opt-in: it puts a finder on sys.meta_path so `mlx_lm.models.<name>` resolves
to Knurlogic's vendored architectures (needed to load deepseek_v4 and
qwen4_exp, which stock mlx-lm lacks). It is idempotent, and
VQLAB_VENDORED_ARCH=0 turns it off. Every `vqlab` CLI command calls it.

See README.md and METHODOLOGY.md.
"""

__version__ = "0.1.0"

from . import _layout  # noqa: F401  stage folders + pre-split import names
from .family import arch as _arch


def use_knurlogic_arch() -> bool:
    """Serve Knurlogic's architectures (deepseek_v4, qwen4_exp, and its
    qwen3_5 / gemma4 / glm5_next) as mlx_lm.models.<name>. Call it before
    mlx_lm.load. Idempotent; returns False when VQLAB_VENDORED_ARCH=0 or
    knurlogic is not installed (stock mlx-lm then serves every name)."""
    return _arch.install()


def _main_is_vqlab() -> bool:
    """True when the running program is one of vqlab's own scripts
    (`python -m vqlab.<stage>.<tool>` or `python <pkg>/<stage>/<tool>.py`):
    those are vqlab's entry points, and they load models as the CLI does."""
    import os
    import sys
    argv = list(getattr(sys, "orig_argv", None) or [])
    for i, a in enumerate(argv[1:], 1):
        if a == "-m":
            nxt = argv[i + 1] if i + 1 < len(argv) else ""
            return nxt == "vqlab" or nxt.startswith("vqlab.")
        if a.startswith("-m"):
            return a[2:] == "vqlab" or a[2:].startswith("vqlab.")
        if a.startswith("-"):
            if a in ("-c",):
                return False
            continue
        pkg = os.path.dirname(os.path.realpath(__file__))
        return os.path.realpath(a).startswith(pkg + os.sep)
    return False


if _main_is_vqlab():
    use_knurlogic_arch()


from .mtp import (
    FAMILIES,
    FamilySpec,
    MTPResponse,
    load_mtp_head,
    mtp_generate,
    mtp_stream_generate,
    register,
)

__all__ = ["use_knurlogic_arch", "FAMILIES", "FamilySpec", "MTPResponse", "load_mtp_head",
           "mtp_generate", "mtp_stream_generate", "register", "__version__"]
