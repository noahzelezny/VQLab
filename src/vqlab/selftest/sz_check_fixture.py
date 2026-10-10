"""The sz-check tool's self-check (was skipzero/sz_check.py `selftest`,
moved here in untangle step 14): a synthetic artifact with dead rows, packed,
expanded (numpy AND mlx-CPU), asserted. CPU only.

    vqlab sz-check --selftest
    python -m vqlab.selftest.sz_check_fixture
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import sys
import tempfile

import numpy as np

from vqlab.skipzero import sz_check as C
from vqlab.skipzero import sz_pack


def _write_st(path, tensors):
    """tensors: {name: np.ndarray} -> an mlx-format safetensors file."""
    dt = {np.dtype(np.uint32): "U32", np.dtype(np.uint8): "U8", np.dtype(np.float16): "F16",
          np.dtype(np.uint16): "U16"}
    specs = [(k, dt[v.dtype], list(v.shape), None) for k, v in tensors.items()]
    hb, _ = sz_pack.header_bytes(specs, {"format": "mlx"})
    with open(path, "wb") as f:
        f.write(hb)
        for v in tensors.values():
            f.write(np.ascontiguousarray(v).tobytes())


def run(root=None):
    """Synthetic artifact with dead rows: pack, expand (numpy AND mlx-CPU),
    assert live rows byte-identical and dead rows exactly zero."""
    rng = np.random.default_rng(1234)
    root = pathlib.Path(root or tempfile.mkdtemp(prefix="szself-"))
    src, out = root / "src", root / "packed"
    src.mkdir(parents=True)
    E, OUT, IN, D, G, PB = 4, 12, 128, 4, 64, 8
    W = (IN // D + 31) // 32 * PB
    p = "model.layers.0.mlp.switch_mlp.up_proj"
    codes = rng.integers(0, 2 ** 32, (E, OUT, W), dtype=np.uint32)
    scales = rng.uniform(0.01, 1.0, (E, OUT, IN // G)).astype(np.float16)
    scales[0, 3] = 0                                   # exact-zero row
    scales[2, 5] = np.float16(1e-6)                    # tiny row
    scales[3] = np.float16(3e-5)                       # whole dead expert
    scales[1, 7, 0] = 0                                # partly zero: LIVE
    cb = rng.standard_normal((256, D)).astype(np.float16)
    other = rng.standard_normal((8, 8)).astype(np.float16)
    _write_st(src / "model-00001-of-00001.safetensors",
              {p + ".codebook": cb, p + ".codes": codes, p + ".vq_scales": scales,
               "model.norm.weight": other})
    (src / "config.json").write_text(json.dumps({"model_type": "synthetic", "model_file": "model.py",
        "vq_modules": {p: {"experts": E, "out": OUT, "in": IN, "k": 256, "dim": D,
                           "group": G, "pack_bits": PB}}}))
    (src / "model.py").write_text("import pathlib as _pathlib\nclass Model:\n"
                                  "    def load_weights(self, w, strict=True):\n"
                                  "        self.w = dict(w)\n")
    (src / "tokenizer.json").write_text("{}")
    assert sz_pack.main([str(src), "--out", str(out), "--allow-any-out"]) == 0
    rep = C.check(out, src)
    assert rep["verdict"] == "PASS", rep["fails"]
    assert json.loads((out / "config.json").read_text())["vq_skipzero"]["modules"][p]["dead_rows"] == OUT + 2
    assert os.path.islink(out / "tokenizer.json")
    # the loader hook, through the generated model.py, on the mlx CPU path
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    spec = importlib.util.spec_from_file_location("sz_model", out / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m = mod.Model()
    w = mx.load(str(out / "model-00001-of-00001.safetensors"))
    m.load_weights(list(w.items()))
    got_c, got_s = np.array(m.w[p + ".codes"]), np.array(m.w[p + ".vq_scales"])
    dead = (scales < 6.2e-5).all(-1)
    assert got_c.dtype == codes.dtype and got_c.shape == codes.shape
    assert np.array_equal(got_c[~dead], codes[~dead]) and not got_c[dead].any()
    assert got_s[~dead].tobytes() == scales[~dead].tobytes() and not got_s[dead].any()
    assert not any(".sz_" in k for k in m.w) and "model.norm.weight" in m.w
    print(f"sz-check selftest PASS ({int(dead.sum())} dead rows of {E * OUT}; "
          f"numpy + mlx-CPU expansion; saved {rep['saved_bytes']} B) in {root}")
    return 0


def main(argv=None):
    return run()


if __name__ == "__main__":
    sys.exit(main())
