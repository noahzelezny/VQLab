"""vqlab alloc-sweep — measure the per-layer allocation frontier for a rung.

THE PROCESS, not a guess. Allocation work on this project repeatedly went
wrong the same way: pick a number of layers to promote or demote, build it,
and generalize from that single point ("12 demotions hurt, so demotion is
bad"). That is n=1 with a conclusion attached. This command produces the two
CURVES the decision actually needs, moving one variable at a time:

  COST  curve — promotion fixed at ZERO, demote the D coldest layers.
                ppl cost per MB saved on the cold end.
  VALUE curve — demotion fixed at ZERO, promote the P hottest layers.
                ppl gain per MB spent on the hot end.

Both anchored on the SAME baseline artifact. Where marginal value meets
marginal cost is the optimal iso-byte allocation — read off the data
instead of chosen by hand.

    vqlab alloc-sweep --artifact <dir> --teacher <bf16> --family qwen4_exp \
        --leverage lev.json [--leverage lev_code.json] --out sweep/ \
        [--cold 0,4,8,12,16,24] [--hot 1,2,4,6,8]

LAYER ORDER comes from `vqlab layer-leverage`, ranked by the JUMP in
`traj_rel` (the per-layer contribution to accumulated drift), NOT by
`local_rel`. Isolation damage is anti-signal for allocation: measuring one
layer against an intact network is the condition where downstream laundering
hides it (quantlab E12 on GLM/affine; F93-F95 on Flash/VQ). Pass --leverage
more than once to average several corpora (they agree closely — r=0.931
prose vs code on Flash-2.1 — so one is usually enough).

Scoring uses the house three corpora through scripts/score_ppl_resident.py
at --score-tokens (default 2048, the published cards' instrument). Ppl is
the GATE; rank with KL if you want a ranking instrument (quantlab III).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REF = os.path.join(REPO, "src", "vqlab", "referee")
CORPORA = {"prose": "referee_corpus.txt",
           "code": "referee_corpus_code_public.txt",
           "lit": "referee_corpus_literary.txt"}
GSZ = 64


def rank_layers(paths):
    """Average the drift-JUMP ranking across one or more leverage maps."""
    acc = {}
    for p in paths:
        r = json.load(open(p))
        rows = r.get("layers", r) if isinstance(r, dict) else r
        tr = {x["layer"]: x["traj_rel"] for x in rows}
        for L in tr:
            acc.setdefault(L, []).append(tr[L] - tr.get(L - 1, 0.0))
    mean = {L: sum(v) / len(v) for L, v in acc.items()}
    return [L for L, _ in sorted(mean.items(), key=lambda kv: -kv[1])], mean


def module_bytes(E, OUT, IN, d, K):
    nsub = IN // d
    return E * OUT * (math.ceil(nsub / 32) * math.ceil(math.log2(K))) * 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--leverage", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cold", default="0,4,8,12,16,24",
                    help="demotion counts for the COST curve")
    ap.add_argument("--hot", default="1,2,4,6,8",
                    help="promotion counts for the VALUE curve")
    ap.add_argument("--base-down", default="4:256", help="baseline down_proj d:K")
    ap.add_argument("--base-gu", default="8:16384", help="baseline gate/up d:K")
    ap.add_argument("--cold-down", default="4:128", help="demoted down_proj d:K")
    ap.add_argument("--hot-gu", default="4:256", help="promoted gate/up d:K")
    ap.add_argument("--protect", default="0,1",
                    help="layers left exactly as shipped (front protection)")
    ap.add_argument("--score-tokens", type=int, default=2048)
    ap.add_argument("--reuse", action="append", default=[])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    protect = {int(x) for x in a.protect.split(",") if x != ""}
    gd, gk = (int(x) for x in a.base_gu.split(":"))
    dd, dk = (int(x) for x in a.base_down.split(":"))
    cd, ck = (int(x) for x in a.cold_down.split(":"))
    hd, hk = (int(x) for x in a.hot_gu.split(":"))

    cfg = json.load(open(os.path.join(a.artifact, "config.json")))
    vm = cfg.get("vq_modules") or cfg.get("vq_linear") or {}
    order, _ = rank_layers(a.leverage)
    order = [L for L in order if L not in protect]

    def geo_for(n_cold, n_hot):
        g = {}
        for L in order:
            g[f"model.layers.{L}.mlp.switch_mlp.down_proj"] = {"dim": dd, "k": dk}
            for p in ("gate_proj", "up_proj"):
                g[f"model.layers.{L}.mlp.switch_mlp.{p}"] = {"dim": gd, "k": gk}
        for L in order[len(order) - n_cold:] if n_cold else []:
            g[f"model.layers.{L}.mlp.switch_mlp.down_proj"] = {"dim": cd, "k": ck}
        for L in order[:n_hot]:
            for p in ("gate_proj", "up_proj"):
                g[f"model.layers.{L}.mlp.switch_mlp.{p}"] = {"dim": hd, "k": hk}
        return g

    def delta_mb(n_cold, n_hot):
        tot = 0
        for L, kind in [(L, "c") for L in order[len(order) - n_cold:] if n_cold] + \
                       [(L, "h") for L in order[:n_hot]]:
            key = (f"model.layers.{L}.mlp.switch_mlp."
                   + ("down_proj" if kind == "c" else "gate_proj"))
            e = vm[key]
            E, OUT, IN = e["experts"], e["out"], e["in"]
            if kind == "c":
                tot += module_bytes(E, OUT, IN, cd, ck) - module_bytes(E, OUT, IN, dd, dk)
            else:
                tot += 2 * (module_bytes(E, OUT, IN, hd, hk)
                            - module_bytes(E, OUT, IN, gd, gk))
        return tot / 1e6

    points = ([(f"cost_D{d}", d, 0) for d in (int(x) for x in a.cold.split(","))]
              + [(f"value_P{p}", 0, p) for p in (int(x) for x in a.hot.split(","))])
    results = {}
    for name, nc, nh in points:
        gm = os.path.join(a.out, f"geomap_{name}.json")
        json.dump(geo_for(nc, nh), open(gm, "w"), indent=1)
        dmb = delta_mb(nc, nh)
        print(f"{name:12} demote={nc:<3} promote={nh:<3} {dmb:+8.1f} MB", flush=True)
        if a.dry_run:
            continue
        art = os.path.join(a.out, f"art_{name}")
        cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "geo_build.py"),
               "--artifact", a.artifact, "--teacher", a.teacher, "--family", a.family,
               "--geomap", gm, "--out", art]
        for r in a.reuse + [os.path.join(a.out, f"art_{n}_parts") for n, _, _ in points]:
            if os.path.isdir(r):
                cmd += ["--reuse", r]
        for attempt in range(60):          # GPU timeouts under disk contention
            if subprocess.run(cmd).returncode == 0:
                break
        row = {"delta_mb": dmb, "demote": nc, "promote": nh}
        for tag, fn in CORPORA.items():
            out = subprocess.run(
                [sys.executable, os.path.join(REPO, "scripts", "score_ppl_resident.py"),
                 "--model", art, "--corpus", os.path.join(REF, fn),
                 "--max-tokens", str(a.score_tokens)],
                capture_output=True, text=True).stdout
            try:
                row[tag] = json.loads(out[out.index("{"):])["ppl"]
            except Exception:
                row[tag] = None
        results[name] = row
        print(f"  -> {row}", flush=True)
        json.dump(results, open(os.path.join(a.out, "sweep_results.json"), "w"), indent=1)

    if results:
        print("\nMARGINAL ppl per 100 MB (the frontier):")
        base = results.get("cost_D0", {})
        for name, r in results.items():
            if r.get("prose") is None or not base.get("prose") or not r["delta_mb"]:
                continue
            print(f"  {name:12} " + "  ".join(
                f"{t}={100*(r[t]-base[t])/abs(r['delta_mb']):+.4f}" for t in CORPORA))


if __name__ == "__main__":
    main()
