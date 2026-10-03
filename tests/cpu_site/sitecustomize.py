"""Put on PYTHONPATH by tests that run writers in a subprocess: every MLX op
in that process runs on the CPU, so the contract tests never touch the GPU
(another session may be benchmarking on it)."""
try:
    import mlx.core as _mx
    _mx.set_default_device(_mx.cpu)
except Exception:  # noqa: BLE001  mlx absent: nothing to pin
    pass
