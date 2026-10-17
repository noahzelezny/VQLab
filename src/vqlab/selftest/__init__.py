"""vqlab selftest — run the real pipeline on a tiny synthetic model.

This is not a mock. It synthesizes a small checkpoint, then runs the SHIPPED
fitter, gate, packer, manifest and kernels over it as subprocesses, exactly
as a user would, and checks the properties each stage is supposed to
guarantee. It needs no downloaded model and finishes in well under a minute.

Every check that can be gated in both directions is (III.5: a gate must FAIL
on a known-bad input and PASS on a known-good one before its pass means
anything). Checks that would need a multi-GB real model — end-to-end
generation through mlx-lm, scoring — are reported as SKIPPED with the reason,
never silently omitted.

    vqlab selftest [--keep] [--verbose]

Layout (untangle step 14; was gate/selftest.py): this file is the runner and
the PASS/FAIL/SKIP reporting. The tiny fixtures are synthetic_model.py; every
check lives in the module for the pipeline stage it exercises (fit_checks,
gate_checks, build_checks, records_checks, kernel_checks, family_checks,
skipzero_checks, plan_checks, queue_checks, layout_checks). The self-checks
that used to sit inside production tools (reselect, sz-check, sz-bitexact)
are reselect_fixture.py, sz_check_fixture.py and sz_bitexact_cpu.py; the
CLI routes those tools' self-check flags here (vqlab.commands.SELFTEST_ROUTES).
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

PY = sys.executable

PASS, FAIL, SKIP = [], [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{(' — ' + detail) if detail else ''}")
    return ok


def skip(name, why):
    SKIP.append(name)
    print(f"  SKIP  {name} — {why}")


def run(args, expect_rc=0, verbose=False):
    p = subprocess.run([PY, *args], capture_output=True, text=True)
    if verbose or (p.returncode != expect_rc):
        print(p.stdout[-2000:], p.stderr[-2000:], sep="\n")
    return p


def report() -> int:
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    return 1 if FAIL else 0


@dataclass
class Ctx:
    """What one stage hands the next: the workspace and the artifacts built
    so far (later stages reuse earlier stages' outputs, as a user would)."""
    tmp: pathlib.Path
    verbose: bool = False
    src: pathlib.Path | None = None      # the synthetic dense checkpoint
    f1: pathlib.Path | None = None       # seeded fit
    f3: pathlib.Path | None = None       # unseeded fit
    A: dict = field(default_factory=dict)   # f1 decoded
    m0: str = ""                         # first module of f1
    packed: pathlib.Path | None = None
    geo: dict = field(default_factory=dict)  # synthetic_model.geo_fixture
    novel: pathlib.Path | None = None    # the unknown-family teacher
    fenv: dict = field(default_factory=dict)
    psrc: pathlib.Path | None = None     # pin source


def _stages():
    from vqlab.selftest import (build_checks, family_checks, fit_checks, gate_checks,
                                kernel_checks, layout_checks, plan_checks, queue_checks,
                                records_checks, skipzero_checks)
    # In the order they print. A stage returning False stops the run (only
    # the first fit can: nothing downstream has an artifact without it).
    return [fit_checks.check_fitter, gate_checks.check_outlier_gate,
            build_checks.check_packer, records_checks.check_manifest,
            records_checks.check_build_records, gate_checks.check_bundle_gate,
            kernel_checks.check_runtime_kernels, fit_checks.check_fit_moe,
            records_checks.check_cli_build_records, family_checks.check_family_profile,
            family_checks.check_onboard, gate_checks.check_pin_and_step_verdict,
            fit_checks.check_reselect, fit_checks.check_additive,
            skipzero_checks.check_skipzero, plan_checks.check_zero_groups,
            family_checks.check_teacher_naming, queue_checks.check_queue,
            layout_checks.check_corpus_locator, gate_checks.check_spelling,
            layout_checks.check_layout, plan_checks.check_pricer,
            plan_checks.check_paper_tables]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vqlab selftest",
                                 description=__doc__.split("\n")[0])
    ap.add_argument("--keep", action="store_true", help="keep the temp workspace")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="vqlab_selftest_"))
    # Every fitter files into the fit store; point it (and the family data
    # dir) at the workspace so a selftest never writes to a real store.
    os.environ["VQLAB_FIT_STORE"] = str(tmp / "fitstore")
    try:
        print("NOTE: this runs REAL Metal kernels and real k-means fits. It is "
              "small (seconds of GPU) but it CONTENDS.\n      Do not run it on a "
              "box that is mid-experiment — strictly sequential is the rule "
              "this project's\n      results depend on.\n")
        print(f"workspace: {tmp}\n")
        from vqlab.selftest import synthetic_model
        ctx = Ctx(tmp=tmp, verbose=a.verbose, src=tmp / "src")
        synthetic_model.make_source(ctx.src)
        for stage in _stages():
            if stage(ctx) is False:
                return report()
        skip("end-to-end generation smoke",
             "needs a real checkpoint + mlx-lm architecture; run "
             "`vqlab smoke` on a real artifact")
        skip("scoring (referee / KL)",
             "needs a real model and teacher cache")
        return report()
    finally:
        if a.keep:
            print(f"\nworkspace kept: {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
