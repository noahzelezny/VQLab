"""III.11 smoke in a CLEAN-ROOM environment — the downloader experience.

Env: fresh venv, `pip install mlx-lm==0.31.3` only. No patch_mlx_lm.py, no
vq_switch.py in site-packages. If this generates a token, the artifact carries
its own runtime and an external user needs nothing from us.
"""
import sys, os, hashlib, time
import mlx.core as mx, mlx_lm
from mlx_lm.utils import load
from mlx_lm import generate

ART = sys.argv[1]
p = os.path.dirname(mlx_lm.__file__)
print("mlx", mx.__version__, "| mlx_lm", mlx_lm.__version__)
print("site-packages vq_switch present:", os.path.exists(p + "/models/vq_switch.py"))
print("utils.py md5:", hashlib.md5(open(p + "/utils.py", "rb").read()).hexdigest())
print("artifact:", ART)
t0 = time.time()
m, t = load(ART)
print("loaded %.0fs" % (time.time() - t0))
# Prove WHICH code is executing. NOTE: do NOT look in sys.modules — mlx_lm
# loads the bundle with module_from_spec + exec_module, which does NOT register
# the module in sys.modules. An earlier version of this probe checked
# sys.modules and printed "the bundle was BYPASSED" on a run where the bundle
# had plainly executed. The authoritative test is the model class's __module__.
cls = type(m).__module__
print("model class module:", cls,
      "<- the artifact's own bundle" if cls == "custom_model"
      else "<- NOT the bundle")
out = generate(m, t, prompt="The capital of France is", max_tokens=8)
print("SMOKE OK:", repr(out))
