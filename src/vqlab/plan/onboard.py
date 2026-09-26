#!/usr/bin/env python3
"""vqlab onboard — sequence a new teacher through docs/ONBOARDING.md.

Every step is an instrument that already exists; this only ORDERS them,
records each result, and resumes. Run it again (or have an agent call it
over MCP) and it picks up where it stopped:

  profile      CPU  inline   vqlab family-profile: arch, legal (d,K), bytes.
                             UNKNOWN family -> stops until the drafted entry
                             is accepted (--accept-entry NAME writes it).
  loader       CPU  inline   core/expert_src loads the first and last VQ
                             layer's stack; shapes must match the profile.
  cache_a/b    GPU  launched `vqlab kl cache` twice, identical inputs.
  determinism  CPU  inline   are the two caches byte-identical? A family
                             whose teacher scores are not deterministic
                             cannot rank quants (ONBOARDING §0; gemma-4).
  init_sweep   GPU  launched `vqlab probe-init` across depth at low K,
                             2 seeds (ONBOARDING §2).

GPU steps launch through the MCP runner (agents/mcp_server.t_run):
detached, under the GPU lease, refused while an exo instance is placed.
Without --launch the next command is printed, not run.

State: families/<family>/teachers/<teacher>/onboard.json. Scratch (caches,
logs): $VQLAB_SCRATCH, default <scratch>/onboard/.

    vqlab onboard --teacher <dir> [--family F] [--launch] [--accept-entry NAME]

After init_sweep the family is characterised; the NEXT move is a decision
(which first rung to build), and `onboard` prints what it knows to make it:
the legal geometries and their exact sizes from the profile.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # src/
from vqlab import _layout  # noqa: E402,F401  one module object per name
import fitstore  # noqa: E402

REPO = _layout.SRC.parent
PY = sys.executable
STEPS = ("profile", "loader", "cache_a", "cache_b", "determinism", "init_sweep")
GPU_STEPS = {"cache_a", "cache_b", "init_sweep"}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def scratch(teacher_slug):
    base = os.environ.get("VQLAB_SCRATCH") or "<scratch>"
    if not pathlib.Path(base).is_dir():
        base = str(pathlib.Path.home() / ".vqlab" / "scratch")
    d = pathlib.Path(base) / "onboard" / teacher_slug
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_tool(args):
    return subprocess.run([PY, "-m", "vqlab.cli", *args], capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(_layout.SRC)})


# ---------------------------------------------------------------- CPU steps
def step_profile(ctx):
    args = ["family-profile", "--teacher", str(ctx["teacher"])]
    if ctx["family"]:
        args += ["--family", ctx["family"]]
    if ctx["accept_entry"]:
        args += ["--write-entry", ctx["accept_entry"]]
    p = run_tool(args)
    if p.returncode:
        return "failed", {"stderr": p.stderr[-800:]}
    out = [l for l in p.stdout.splitlines() if l.startswith("-> ")][-1][3:]
    prof = json.loads(pathlib.Path(out).read_text())
    if prof.get("unknown_family"):
        return "blocked", {
            "why": "no families entry reads this checkpoint",
            "suggested_entry": prof.get("suggested_entry"),
            "next": "review the drafted entry, then rerun with --accept-entry <name>"}
    ctx["family"] = prof["family"]
    ctx["profile"] = prof
    return "done", {"profile": out, "family": prof["family"],
                    "modules": prof["modules"]["count"],
                    "vq_target_gib_per_bit": prof["bytes"]["vq_target_gib_per_bit"]}


def step_loader(ctx):
    import families
    import mlx.core as mx
    prof = ctx["profile"]
    fam = families.FAMILY.get(ctx["family"])
    if fam is None:                       # dense families read by key template
        return "done", {"note": "dense family: key template checked by the profile"}
    import expert_src
    T = ctx["teacher"]
    ip = T / "model.safetensors.index.json"
    if ip.exists():
        idx = json.loads(ip.read_text())["weight_map"]
    else:                                  # single-file checkpoint: no index
        from family_profile import header
        idx = {k: f.name for f in sorted(T.glob("*.safetensors"))
               for k in header(os.path.realpath(f))}
    layers = prof["arch"]["vq_layers"]
    checked = []
    for li in (layers[0], layers[-1]):
        for proj in fam["proj"]:
            with mx.stream(mx.cpu):
                W = expert_src.load_expert_stack(T, idx, fam, li, proj, experts=2)
                mx.eval(W)
            sig = [list(s) for s in prof["modules"]["signatures"][proj]]
            ok = any(W.shape[1:] == tuple(s[1:]) for s in sig)
            checked.append({"layer": li, "proj": proj, "shape": list(W.shape), "ok": ok})
            if not ok:
                return "failed", {"why": "loaded shape disagrees with the profile",
                                  "checked": checked}
    return "done", {"checked": checked}


def _tree_digest(d):
    """sha256 of every non-meta file in a cache dir, by name."""
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(pathlib.Path(d).iterdir())
            if f.is_file() and f.name != "meta.json"}


def step_determinism(ctx):
    a, b = (ctx["state"]["steps"][s]["result"].get("out_dir") for s in ("cache_a", "cache_b"))
    da, db = _tree_digest(a), _tree_digest(b)
    same = bool(da) and da == db
    return "done", {"deterministic": same, "files": sorted(da),
                    "verdict": ("teacher scores are deterministic: this family can be "
                                "ranked" if same else
                                "NOT deterministic: quants of this family cannot be "
                                "ranked on this path (see ONBOARDING §0)")}


# ---------------------------------------------------------------- GPU steps
def gpu_command(step, ctx):
    T, sc = ctx["teacher"], ctx["scratch"]
    if step in ("cache_a", "cache_b"):
        if streamed_scorer(ctx):
            # The KL GATE's own instrument (kl-ladder scores through it), not
            # kl_damage: it streams layers, so the teacher need not fit
            # resident, and it loads on the family's validated path
            # (cpu_stream_load where F120 needs it). kl_damage's resident
            # load of the 35B teacher off the HDD tripped the GPU watchdog
            # in 12 s on 2026-09-26.
            return ["stream-score", "--model", str(T), "--corpus", str(_layout.corpus("prose")),
                    "--tokens", "2048", "--chunk", "512", "--save-topk", "64",
                    "--out", str(sc / step)], {"out_dir": str(sc / step), "instrument": "stream-score"}
        return ["kl", "cache", "--model", str(T), "--out-dir", str(sc / step),
                "--seed", "1234"], {"out_dir": str(sc / step), "instrument": "kl cache"}
    if step == "init_sweep":
        L = ctx["profile"]["arch"]["vq_layers"]
        pick = sorted({L[0], L[len(L) // 3], L[2 * len(L) // 3], L[-1]})
        return ["probe-init", "--src", str(T), "--family", ctx["family"],
                "--layers", ",".join(map(str, pick)), "--k", "256", "--dim", "4",
                "--reps", "2"], {}
    raise KeyError(step)


def streamed_scorer(ctx) -> bool:
    """Does stream_score have a VALIDATED scorer for this teacher's model_type?"""
    arch = (ctx.get("profile") or {}).get("arch") or {}
    from vqlab.score import stream_score
    e = stream_score.SCORERS.get(arch.get("model_type")) or \
        stream_score.SCORERS.get(arch.get("text_model_type"))
    return bool(e and e.get("validated"))


def fits_resident(ctx):
    p = run_tool(["preflight-ram", str(ctx["teacher"])])
    return p.returncode == 0, (p.stdout + p.stderr).strip().splitlines()[-1:] or [""]


def launch(cmd_args, tag):
    import importlib
    ms = importlib.import_module("vqlab.agents.mcp_server")
    return ms.t_run(cmd_args[0], cmd_args[1:], tag=tag)


def poll(run_id):
    import importlib
    ms = importlib.import_module("vqlab.agents.mcp_server")
    st = ms.t_status(run_id, tail=20)
    s = st.get("status")
    if s == "completed":
        return ("done" if st.get("exit_code") == 0 else "failed"), st
    if s in ("failed", "deferred", "stopped"):
        return "failed" if s != "deferred" else "pending", st
    return "running", st


# ------------------------------------------------------------------- driver
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vqlab onboard", description=__doc__.split("\n")[0])
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--family")
    ap.add_argument("--launch", action="store_true",
                    help="launch the next GPU step through the MCP runner (lease-gated)")
    ap.add_argument("--accept-entry", metavar="NAME",
                    help="unknown family: accept the drafted entry as families/NAME")
    ap.add_argument("--redo", metavar="STEP", choices=STEPS, help="reset one step")
    a = ap.parse_args(argv)

    T = pathlib.Path(a.teacher)
    slug = fitstore.teacher_slug(T)
    ctx = {"teacher": T, "family": a.family, "accept_entry": a.accept_entry,
           "scratch": scratch(slug)}

    # the profile step decides the family, which decides where state lives
    st, res = step_profile(ctx)
    fam = ctx["family"] or "_unknown"
    fams = pathlib.Path(os.environ.get("VQLAB_FAMILIES_DIR") or REPO / "families")
    sp = fams / fam / "teachers" / slug / "onboard.json"
    sp.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(sp.read_text()) if sp.exists() else {
        "schema": "vqlab.onboard/1", "teacher": str(T), "slug": slug,
        "steps": {s: {"status": "pending", "result": {}} for s in STEPS}}
    state["family"] = fam
    ctx["state"] = state
    if a.redo:
        state["steps"][a.redo] = {"status": "pending", "result": {}}
    state["steps"]["profile"] = {"status": st, "result": res, "at": now()}

    def save():
        state["updated"] = now()
        sp.write_text(json.dumps(state, indent=1, default=str))

    for step in STEPS[1:]:
        s = state["steps"][step]
        if s["status"] in ("done", "blocked"):
            continue
        prev = state["steps"][STEPS[STEPS.index(step) - 1]]["status"]
        if prev not in ("done", "blocked"):
            break                       # strictly sequential: one step at a time
        if step == "determinism" and any(
                state["steps"][x]["status"] == "blocked" for x in ("cache_a", "cache_b")):
            s["status"] = "blocked"
            s["result"] = {"why": "no teacher caches on this box (see cache_a)"}
            save()
            continue
        if step not in GPU_STEPS:
            try:
                s["status"], s["result"] = step_determinism(ctx) \
                    if step == "determinism" else step_loader(ctx)
            except Exception as e:     # a crashed step is a recorded failure
                import traceback
                s["status"] = "failed"
                s["result"] = {"error": f"{type(e).__name__}: {e}",
                               "trace": traceback.format_exc()[-1500:]}
            s["at"] = now()
            save()
            if s["status"] != "done":
                break
            continue
        if s["status"] == "running":
            s["status"], info = poll(s["result"]["run_id"])
            s["result"]["log_tail"] = info.get("log_tail", "")[-1500:]
            save()
            if s["status"] != "done":
                break
            continue
        if step in ("cache_a", "cache_b") and not streamed_scorer(ctx):
            ok, why = fits_resident(ctx)
            if not ok:
                s["status"] = "blocked"
                s["result"] = {"why": "teacher does not fit resident on this box; "
                                      "kl cache loads it whole. Build the cache on a "
                                      "bigger box or with the streamed scorer "
                                      "(vqlab score / stream_score)", "preflight": why}
                save()
                continue
        cmd, extra = gpu_command(step, ctx)
        if not a.launch:
            s["result"] = {**extra, "command": "vqlab " + " ".join(cmd)}
            save()
            print(f"next: {step}  ->  vqlab {' '.join(cmd)}\n"
                  f"      (rerun with --launch to start it under the GPU lease)")
            break
        try:
            r = launch(cmd, f"onboard-{step}")
        except Exception as e:
            s["result"] = {**extra, "launch_refused": str(getattr(e, "message", e))}
            save()
            print(f"{step}: launch refused: {s['result']['launch_refused']}")
            break
        s["status"] = "running"
        s["result"] = {**extra, "run_id": r["run_id"], "command": r["command"]}
        s["at"] = now()
        save()
        print(f"{step}: launched {r['run_id']}")
        break
    save()

    print(f"\nonboard {slug} ({fam}) -> {sp}")
    for step in STEPS:
        s = state["steps"][step]
        r = s.get("result") or {}
        note = (r.get("verdict") or r.get("why") or r.get("run_id")
                or r.get("launch_refused") or r.get("command") or "")
        print(f"  {s['status']:8s} {step:12s} {str(note)[:100]}")
    if all(state["steps"][x]["status"] in ("done", "blocked") for x in STEPS) and \
            state["steps"]["profile"]["status"] == "done":
        prof = ctx["profile"]
        print("\ncharacterised. First-rung options (legal geometries, exact GiB for "
              "the whole VQ target):")
        for proj, gs in prof["geometries"].items():
            ok = [g for g in gs if g["legal"]]
            print(f"  {proj}: " + ", ".join(
                f"d{g['d']}-K{g['K']} {g['stack_gib']}GiB" for g in ok[:8]))
        print("  price it (vqlab price), fit it (fit-moe / fit-dense, which file into "
              "the store), then vqlab layer-leverage against the teacher.")
    return 0 if state["steps"]["profile"]["status"] == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
