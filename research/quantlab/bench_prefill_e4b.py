"""E147 — long-prompt prefill, VQ-PLE vs 8-bit incumbent.
Same instrument as the card's number (stream_generate prompt_tps) so it is
comparable, but at lengths where prefill dominates instead of per-call overhead.
Pre-registered: the ~20% gap should NARROW or VANISH with length. If it WIDENS
at 8k that is a real VQ-path throughput difference — report it, do not smooth it.
"""
import sys, time, statistics, json
import mlx.core as mx
from mlx_lm.utils import load
from mlx_lm import stream_generate

E = "/Volumes/Thunderbay SSD/Exo Models"
MODELS = [("VQ-PLE", f"{E}/e4b-VQ-pleonly-packed"),
          ("8bit",   f"{E}/mlx-community--gemma-4-e4b-it-8bit")]
LENGTHS = [30, 2048, 8192]
REPS = 3

corpus = open("referee/referee_corpus.txt").read()
out = {}
for label, path in MODELS:
    model, tok = load(path)
    ids_all = tok.encode(corpus)
    out[label] = {}
    for L in LENGTHS:
        if len(ids_all) < L:
            print(f"  !! corpus only {len(ids_all)} tokens, need {L}", flush=True)
            continue
        prompt = tok.decode(ids_all[:L])
        n_tok = len(tok.encode(prompt))
        runs = []
        for r in range(REPS + 1):          # first is warmup, discarded
            mx.clear_cache()
            resp = None
            for resp in stream_generate(model, tok, prompt=prompt, max_tokens=4):
                pass
            if r:
                runs.append((resp.prompt_tps, getattr(resp, "peak_memory", float("nan"))))
        tps = [x[0] for x in runs]
        out[label][L] = dict(prompt_tokens=n_tok,
                             prompt_tps_median=statistics.median(tps),
                             prompt_tps_runs=[round(t, 1) for t in tps],
                             peak_gb=round(runs[-1][1], 3))
        print(f"  {label:<7} L={L:<5} tokens={n_tok:<5} prompt_tps median {statistics.median(tps):8.1f} "
              f"runs {[round(t,1) for t in tps]}  peak {runs[-1][1]:.2f} GB", flush=True)
    del model, tok
    mx.clear_cache()

print("\n=== RATIO (8bit / VQ-PLE); >1 means VQ is slower ===")
for L in LENGTHS:
    if L in out.get("VQ-PLE", {}) and L in out.get("8bit", {}):
        v = out["VQ-PLE"][L]["prompt_tps_median"]; a = out["8bit"][L]["prompt_tps_median"]
        print(f"  L={L:<5} VQ {v:8.1f}  8bit {a:8.1f}   VQ is {100*(a-v)/a:5.1f}% slower")
print("\nCARD CLAIM: ~20% slower, from 392 vs 496 tok/s at ~30 tokens.")
print("PRE-REG: gap should NARROW or VANISH at 2k/8k. Widening = real VQ-path cost.")
json.dump(out, open("results_e147_prefill.json", "w"), indent=1)
