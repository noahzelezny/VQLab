#!/usr/bin/env python3
"""Standalone referee: place a model on the exo cluster, score it, evict.

Runs with NOTHING but python3 stdlib + a running exo cluster — no Scout, no
Claude, no venv. Made for the max-RAM sessions (e.g. the 209G 4-bit 397B)
where every other app on both boxes is closed.

Usage (on the M3, with both exo nodes up):

    python3 score_via_exo.py mlx-community/Qwen3.5-397B-A17B-4bit

    # useful knobs:
    #   --cap "Noah's Mac Studio=88,NozzleBook Pro=118"   per-node GiB caps
    #   --runners 2          require 2-node placement (default 2)
    #   --list               just print previews + per-node GiB and exit
    #   --keep               don't evict after scoring

Before a tight fit, raise wired limits (needs sudo, survives until reboot):
    M3:  sudo sysctl iogpu.wired_limit_mb=92160    # 90 GiB of 96
    M4:  sudo sysctl iogpu.wired_limit_mb=122880   # 120 GiB of 128
and set the --cap values a few GiB under those.

The corpus is the SAME 60k-char wikitext slice as every E17 number
(referee_corpus.txt, sha256 81ea5b79…) — scores are directly comparable:
  t2.1-revexp 9.106 · spicyneuron-2.6bit 13.026 · t2.4 18.948 · t2.6 23.980
"""
import argparse
import json
import pathlib
import sys
import time
import urllib.request

API = "http://localhost:52415"


def get(path, timeout=30):
    return json.load(urllib.request.urlopen(API + path, timeout=timeout))


def post(path, payload, timeout=3600):
    req = urllib.request.Request(
        API + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def delete(path):
    req = urllib.request.Request(API + path, method="DELETE")
    return urllib.request.urlopen(req, timeout=60).read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--runners", type=int, default=2)
    ap.add_argument("--cap", default="",
                    help="per-node GiB caps: 'FriendlyName=GiB,Name=GiB'")
    ap.add_argument("--node", default="",
                    help="for --runners 1: require this friendlyName (or "
                        "substring). exo happily offers a single-runner "
                        "preview on a node that doesn't even have the model "
                        "files locally, then loads forever — pin it when "
                        "the model isn't mirrored to every box.")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--sharding", choices=["Tensor", "Pipeline", "any"],
                    default="Tensor",
                    help="which multi-node split to place. Tensor was the "
                         "default for every E17 number, but raw 397B scores "
                         "do NOT reproduce under it (4.824 vs 3.500 same cmd) "
                         "while 35B single-node is exact — suspicion is the "
                         "cross-rank logit gather under Tensor sharding.")
    ap.add_argument("--raw", action="store_true",
                    help="score the BARE corpus (standard wikitext PPL) via "
                         "the <|RAW_SCORE|> sentinel, instead of the chat-"
                         "wrapped number exo produces by default. Needs the "
                         "2026-08-12 utils_mlx.py patch on every node.")
    args = ap.parse_args()

    caps = {}
    for part in filter(None, args.cap.split(",")):
        name, gb = part.rsplit("=", 1)
        caps[name.strip()] = float(gb)

    state = get("/state")
    names = {k: (v.get("friendlyName") or k[:12])
             for k, v in state["nodeIdentities"].items()}
    print(f"nodes up: {list(names.values())}")
    if len(names) < args.runners:
        sys.exit(f"need {args.runners} nodes, have {len(names)} — start exo "
                 "on the other box and wait ~30s for previews to settle")

    prev = get(f"/instance/previews?model_id={args.model}", timeout=60)
    picked = None
    for i, p in enumerate(prev.get("previews", [])):
        inst = (p.get("instance") or {}).get("MlxRingInstance")
        if not inst:
            continue
        sa = inst["shardAssignments"]
        n_run = len(sa["runnerToShard"])
        sharding = "Tensor" if "Tensor" in json.dumps(p["instance"]) else "Pipeline"
        deltas = {}
        for node, gb in (p.get("memory_delta_by_node") or {}).items():
            deltas[names.get(node, node[:12])] = round(gb / 1024**3, 1) \
                if gb > 1e6 else round(gb, 1)  # bytes vs GiB, be tolerant
        fits = all(deltas.get(n, 0) <= caps.get(n, 1e9) for n in deltas) \
            if caps else True
        on_node = (not args.node) or any(
            args.node.lower() in n.lower() for n in deltas)
        tag = ""
        want = (args.sharding == "any" or sharding == args.sharding
                or args.runners == 1)
        if n_run == args.runners and fits and on_node and want and picked is None:
            picked, tag = (i, p), "  <-- PICK"
        print(f"  [{i}] {sharding} runners={n_run} mem/node={deltas}{tag}")
    if args.list:
        return
    if picked is None:
        sys.exit("no preview fits — lower caps are excluding everything, or "
                 "no Tensor placement exists; re-run with --list to inspect, "
                 "and check wired limits were raised")

    i, p = picked
    print(f"creating instance from preview {i}…")
    post("/instance", {"instance": p["instance"]}, timeout=120)

    print("waiting for load (11-min SMB loads are normal; timeout 40 min)…")
    t0 = time.time()
    inst_id = None
    while time.time() - t0 < 2400:
        st = get("/state")
        for iid, inst in (st.get("instances") or {}).items():
            if args.model.lower() in json.dumps(inst).lower():
                inst_id = iid
        blob = json.dumps(st)
        if inst_id and '"RunnerFailed"' in blob:
            sys.exit("runner FAILED during load — usually OOM: close more "
                     "apps / raise wired limit / pick a smaller placement")
        try:
            r = post("/v1/chat/completions", {
                "model": args.model, "max_tokens": 1, "temperature": 0,
                "messages": [{"role": "user", "content": "hi"}]}, timeout=20)
            if r.get("choices"):
                break
        except Exception:
            pass
        time.sleep(15)
    else:
        sys.exit("timed out waiting for the model to answer")
    print(f"loaded + answering after {time.time() - t0:.0f}s; scoring…")

    corpus = (pathlib.Path(__file__).parent / "referee_corpus.txt").read_text()
    r = post("/v1/chat/completions", {
        "model": args.model,
        "messages": [{"role": "user",
                      "content": ("<|RAW_SCORE|>" + corpus) if args.raw
                                 else corpus}],
        "echo_score": True, "max_tokens": 1, "temperature": 0,
    })
    content = r["choices"][0]["message"].get("content") or ""
    try:
        score = json.loads(content)
        print(json.dumps({"model": args.model, **score}, indent=1))
        print(f"\n*** PPL = {score.get('ppl'):.3f} *** "
              f"(t2.1-revexp = 9.106 on this exact corpus)")
    except Exception:
        print("RAW:", content[:400])

    if not args.keep and inst_id:
        print("evicting…")
        delete(f"/instance/{inst_id}")
        # Wait for the instance to actually disappear before returning, or a
        # back-to-back score sees the memory still held and gets "no preview
        # fits". Evicting can also strand a runner subprocess holding ~75G —
        # it is NOT killed here (that needs a signal on the owning node), but
        # the wait at least makes the failure legible instead of mysterious.
        for _ in range(24):
            time.sleep(5)
            try:
                if inst_id not in (get("/state").get("instances") or {}):
                    break
            except Exception:
                break
        else:
            print("  ⚠️  instance still listed after 2 min — if the next "
                  "placement fails, check for an orphaned runner "
                  "(pkill -f multiprocessing.spawn on the owning node)")


if __name__ == "__main__":
    main()
