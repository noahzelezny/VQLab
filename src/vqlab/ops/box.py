"""gpu_state: who holds this box's GPU lease, which exo instances are placed,
every box's reservation, and (with [boxes.*] configured) the same for the
other boxes over ssh."""

from __future__ import annotations

import json
import shlex
import subprocess
from typing import Any, Dict

from vqlab import config
from vqlab.core.box.exo import exo_instances as _exo_instances
from vqlab.core.box.gpu_lease import lease_holder as _lease_holder
from vqlab.core.box.gpu_lease import lease_path
from vqlab.ops.runs import _lab_state, _this_host
from vqlab.queue import reserve as _resv


def _box_state(name: str, b: Dict[str, Any]) -> Dict[str, Any]:
    """Another box's lease holder and GPU-heavy processes, over ssh, from its
    own clone and config (the [boxes.NAME] table). Never raises; a box with
    no `repo` (reachable, but no vqlab clone there) says so."""
    if not b.get("repo") or not b.get("config"):
        return {"error": f"[boxes.{name}] names no vqlab clone (repo, config)"}
    remote = (f"cd {shlex.quote(b['repo'])} && VQLAB_CONFIG={shlex.quote(b['config'])} "
              f"PYTHONPATH=src {shlex.quote(b['python'])} -c "
              + shlex.quote("import json; from vqlab.core.box.gpu_lease import lease_holder; "
                            "from vqlab.bench import box_quiet as q; s=q.state(); "
                            "print(json.dumps({'lease_holder': lease_holder(), 'load1': s['load1'], "
                            "'heavy': s['heavy'][:5]}))"))
    try:
        r = subprocess.run(["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", b["ssh"], remote],
                           capture_output=True, text=True, timeout=30)
        return json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else \
            {"error": (r.stderr or r.stdout).strip()[-300:]}
    except Exception as e:
        return {"error": str(e)[:300]}


def gpu_state(boxes: bool = True) -> Dict[str, Any]:
    out = {"host": config.short_host(), "lease": str(lease_path()),
           "lease_holder": _lease_holder(), "exo_instances": _exo_instances(),
           "lab_state": _lab_state(), "this_host": _this_host()}
    try:   # every box's reservation (`vqlab reserve`), from the shared file
        out["reservations"] = [dict(r, text=_resv.describe(r)) for r in _resv.active()]
        out["reservations_file"] = str(_resv.shared_path())
    except Exception as e:
        out["reservations"] = {"error": str(e)[:300]}
    if boxes and config.boxes():
        out["boxes"] = {n: _box_state(n, b) for n, b in config.boxes().items()}
    return out
