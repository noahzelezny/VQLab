"""DEPRECATED (untangle step 12, 2026-10-10): the queue runner is the
`vqlab.queue` package now. Import from there; this name is kept for one
release.

    vqlab.queue.runner  run loop, one step, the `vqlab queue` CLI (main)
    vqlab.queue.steps   step schema, preflight, input-path and pin checks
    vqlab.queue.files   queue dirs, state.json, create
    vqlab.queue.report  status / list / wait
    vqlab.queue.remote  --on BOX

Every public name this module had, and the private ones tests used, are
re-exported below. They are the SAME objects, but this is a separate module:
monkeypatching a name HERE does not reach the code in vqlab.queue (patch it
where it is used).
"""
import subprocess  # noqa: F401  (tests patched run_queue.subprocess)
import sys
import warnings

from vqlab.gate import pin as pinmod  # noqa: F401
from vqlab.queue import eta as queue_eta  # noqa: F401
from vqlab.queue import reserve as resv  # noqa: F401
from vqlab.records import step_verdict as sv  # noqa: F401
from vqlab.queue.files import (REPO, SCHEMA, _now, _slug, create, installed,  # noqa: F401
                               latest as _latest, package_hash as _package_hash,
                               git as _git, queues_dir, save_state as _save)
from vqlab.queue.remote import (_REMOTE_PROBE, _mount, remote_check, remote_paths,  # noqa: F401
                                run_on)
from vqlab.queue.report import TERMINAL, status, wait  # noqa: F401
from vqlab.queue.runner import _stop, _term, main, run  # noqa: F401
from vqlab.queue.runner import run_step as _run_step  # noqa: F401
from vqlab.queue.steps import (OUT_FLAGS, PREFLIGHT_MARGIN, SCORING, _NAMED,  # noqa: F401
                               _PATHLIKE, pin_check, preflight_root, static_check,
                               validate)
from vqlab.queue.steps import argv as _argv  # noqa: F401
from vqlab.queue.steps import env as _env  # noqa: F401
from vqlab.queue.steps import input_bytes as _input_bytes  # noqa: F401
from vqlab.queue.steps import needs_gpu as _needs_gpu  # noqa: F401
from vqlab.queue.steps import path_args as _path_args  # noqa: F401
from vqlab.queue.steps import redirect_outs as _redirect_outs  # noqa: F401
from vqlab.queue.steps import strip_name as _strip_name  # noqa: F401

warnings.warn("vqlab.agents.run_queue is deprecated; import vqlab.queue.runner "
              "(and .steps / .files / .report / .remote)", DeprecationWarning, stacklevel=2)

if __name__ == "__main__":
    sys.exit(main())
