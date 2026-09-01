#!/bin/sh
cd /Users/noahzelezny/Documents/AgenicAI/quantlab
V=./venv/bin/python
E="/Volumes/Thunderbay SSD/Exo Models"
A="$E/e4b-VQ-pleonly-packed"
echo "== 1. packed KL identity (must be 7.451 / 95.70) =="
$V kl_damage.py score --model "$A" --cache-dir "$E/kl_cache_e4b_LIT" 2>&1 | tail -2
echo "== 2. decode speed =="
$V - <<'PY'
import time, mlx.core as mx
from mlx_lm.utils import load
from mlx_lm import stream_generate
m,t=load("/Volumes/Thunderbay SSD/Exo Models/e4b-VQ-pleonly-packed")
p=t.apply_chat_template([{"role":"user","content":"Write one vivid paragraph describing a harbour town at first light."}],add_generation_prompt=True,tokenize=False,enable_thinking=False)
for r in stream_generate(m,t,prompt=p,max_tokens=8): pass
for r in stream_generate(m,t,prompt=p,max_tokens=160): last=r
print(f"decode {last.generation_tps:.1f} tok/s  prompt {last.prompt_tps:.0f} tok/s  peak {last.peak_memory:.1f}GB")
PY
echo "== 3. litbench cyclic generative =="
$V litbench_chat.py --model "$A" --cyclic --generative \
   --out results_literary/gencyc_e4b-VQple.json 2>&1 | tee -a logs_live_pleonly.log | tail -2
echo "== 4. paired McNemar vs the 8-bit incumbent =="
$V paired_litbench.py results_literary/gencyc_e4b-VQple.json results_literary/gencyc_e4b-8bit.json
echo "########## PLE-ONLY GATES DONE"
