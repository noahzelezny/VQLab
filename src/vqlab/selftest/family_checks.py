"""Selftest: onboarding a family (family-profile, onboard, teacher naming)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess

from vqlab._layout import find as _find
from vqlab.selftest import PY, check
from vqlab.selftest.synthetic_model import novel_teacher


def check_family_profile(ctx):
    print("[6a/7] family-profile: unknown family -> data entry, no code change")
    tmp = ctx.tmp
    ctx.novel = U = tmp / "novel-teacher"
    novel_teacher(U)
    ctx.fenv = fenv = {**os.environ, "VQLAB_FAMILIES_DIR": str(tmp / "fams")}
    fp_ = [PY, str(_find("family_profile.py")), "--teacher", str(U),
           "--out", str(tmp / "novel-prof")]
    p = subprocess.run(fp_, capture_output=True, text=True, env=fenv)
    check("unknown family is reported UNKNOWN, not mis-matched (known-bad)",
          "UNKNOWN" in p.stdout and json.load(open(
              tmp / "novel-prof" / "profile.json")).get("unknown_family") is True,
          p.stdout.splitlines()[0] if p.stdout else p.stderr[-200:])
    p = subprocess.run(fp_ + ["--write-entry", "novelmoe"], capture_output=True,
                       text=True, env=fenv)
    p = subprocess.run(fp_, capture_output=True, text=True, env=fenv)
    prof = json.load(open(tmp / "novel-prof" / "profile.json"))
    check("drafted data entry makes the family profile: 6 modules found",
          prof["family"] == "novelmoe" and prof["modules"]["count"] == 6,
          f"family={prof['family']} modules={prof['modules']['count']}")


def check_onboard(ctx):
    print("[6a2/7] onboard: sequences the steps, resumes, never fakes a GPU step")
    tmp = ctx.tmp
    ob = [PY, str(_find("onboard.py")), "--teacher", str(ctx.novel)]
    oenv = {**ctx.fenv, "VQLAB_SCRATCH": str(tmp / "scratch")}
    p = subprocess.run(ob, capture_output=True, text=True, env=oenv)
    stf = tmp / "fams" / "novelmoe" / "teachers" / "novel-teacher" / "onboard.json"
    ost = json.load(open(stf))["steps"] if stf.exists() else {}
    check("onboard runs the CPU steps and stops at the first GPU step",
          ost.get("profile", {}).get("status") == "done"
          and ost.get("loader", {}).get("status") == "done"
          and ost.get("cache_a", {}).get("status") == "pending"
          and "kl cache" in ost.get("cache_a", {}).get("result", {}).get("command", ""),
          (p.stdout or p.stderr)[-200:])
    p = subprocess.run(ob, capture_output=True, text=True, env=oenv)
    ost2 = json.load(open(stf))["steps"]
    check("rerunning onboard resumes (profile/loader stay done, nothing launched)",
          ost2["loader"]["status"] == "done" and ost2["cache_a"]["status"] == "pending"
          and "run_id" not in ost2["cache_a"]["result"])


def check_teacher_naming(ctx):
    print("[6a35/7] teacher naming by content (VL4.11)")
    from vqlab.core.source import fit_store as FS
    tmp, psrc = ctx.tmp, ctx.psrc
    tfam = tmp / "fam_ident"
    tprof = tfam / "fx" / "teachers" / "Org--Real-Teacher"
    tprof.mkdir(parents=True)
    csha, sfp = FS.teacher_identity(psrc)
    (tprof / "profile.json").write_text(json.dumps(
        {"identity": {"config_sha256": csha, "shard_fingerprint": sfp}}))
    tcopy = tmp / "scratch_copy_of_teacher"
    shutil.copytree(psrc, tcopy)
    check("a byte-identical teacher COPY files under the profiled teacher's name",
          FS.teacher_slug(tcopy, tfam) == "Org--Real-Teacher")
    (tcopy / "config.json").write_text((tcopy / "config.json").read_text() + " ")
    check("a teacher that differs keeps its own directory name",
          FS.teacher_slug(tcopy, tfam) == "scratch_copy_of_teacher")
