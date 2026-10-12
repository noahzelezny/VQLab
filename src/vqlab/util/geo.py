"""The name a VQ geometry goes by wherever a person reads it."""


def geo_key(d, k) -> str:
    """dD-KK, e.g. d4-K256 (codebook vector dim d, K codewords). Not fit-moe's
    own --tail-geom spelling (d4k256), nor a display with padding."""
    return f"d{d}-K{k}"
