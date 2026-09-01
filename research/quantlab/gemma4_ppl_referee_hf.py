#!/usr/bin/env python
"""Independent referee for the gemma-4 ppl anomaly: HF transformers, no mlx.

THE QUESTION THIS SETTLES. Every gemma-4 checkpoint scored through the mlx
ports shows wildly inflated perplexity on external text (plain English ~27-97,
Austen 700-60k) while generating coherently and ranking hellaswag above
chance. Both available mlx implementations (mlx_lm 0.31.3 gemma4,
mlx_vlm 0.5.0 gemma4) produce IDENTICAL numbers to the decimal, so they share
math and cannot referee each other. Two live hypotheses:

  A) Model property: gemma-4-it is RL-sharpened to the point that raw-text
     loglikelihood is meaningless. Then transformers reproduces the inflated
     ppl, and the fix is instrumentation (no raw-ppl benchmarks for gemma),
     not code.
  B) Shared port bug: some subtlety (proportional rope, RMSNormNoScale,
     shared-KV threading, attn_output_gate, per-expert scale...) is wrong in
     the mlx lineage. Then transformers gives sane ppl (~10-25 for a 2B on
     this text) and the ports need a diff-hunt.

Run on mlx-community/gemma-4-e2b-it-bf16 (unquantized, HF tensor layout, and
the SAME repo family the mlx numbers came from, so quantization damage is
excluded by construction). CPU is fine at this size; slow is fine — this is
a referee, not a benchmark.

    ~/Documents/AgenicAI/.venv/bin/python gemma4_ppl_referee_hf.py

Prints ppl for the same three probes used on the mlx side. Compare against:
    e2b-6bit  (mlx_vlm, production sidecar): plain 96.62   austen 729.15
    31b-8bit  (mlx_lm):                      plain 27.15   austen 10449.42
"""
import math
import pathlib
import sys

import torch

SNAP = sorted(pathlib.Path(
    "/Volumes/Thunderbay SSD/Mlx_Models/hub/"
    "models--mlx-community--gemma-4-e2b-it-bf16/snapshots").glob("*"))[-1]

PLAIN = ("The weather in Paris is generally mild in the spring. Tourists "
         "often visit the city in April and May, when the gardens are in "
         "bloom and the cafes put their tables out on the pavement.")
AUSTEN = pathlib.Path(
    "referee/referee_corpus_literary.txt").read_text()[:300]


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"loading {SNAP} (CPU, bf16->fp32)...", flush=True)
    tok = AutoTokenizer.from_pretrained(SNAP)
    model = AutoModelForCausalLM.from_pretrained(
        SNAP, dtype=torch.float32)
    model.eval()

    def ppl(text):
        ids = tok(text, return_tensors="pt").input_ids
        if ids[0, 0].item() != tok.bos_token_id:
            ids = torch.cat(
                [torch.tensor([[tok.bos_token_id]]), ids], dim=1)
        with torch.no_grad():
            out = model(ids)
        lg = out.logits[0, :-1].float()
        tgt = ids[0, 1:]
        lse = torch.logsumexp(lg, dim=-1)
        picked = lg.gather(-1, tgt[:, None])[:, 0]
        return round(math.exp((lse - picked).mean().item()), 2)

    print(f"transformers {sys.modules['transformers'].__version__}")
    print(f"plain english : {ppl(PLAIN)}")
    print(f"austen 300ch  : {ppl(AUSTEN)}")
    print("\nCompare mlx: e2b-6bit plain 96.62 / austen 729.15.")
    print("Sane (≈10-25 plain) -> mlx port bug. Matching inflation -> model "
          "property; drop raw-ppl instruments for gemma-4.")


if __name__ == "__main__":
    main()
