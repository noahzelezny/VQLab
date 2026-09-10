---
language:
- en
- zh
license: mit
library_name: mlx
pipeline_tag: image-text-to-text
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
tags:
- mlx
- quantized
- vector-quantization
- apple-silicon
- glm
---

# TheDrainFlorist/GLM-5.3-Flash-VQ-2.7bpw

**101.9 GiB — GLM-5.3-Flash on one 128 GB Mac.**

A data-free vector-quantized build of
[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (598.5 GiB
bf16, vision) for Apple Silicon. Stock `mlx-vlm`, no patches — the VQ
runtime ships inside the checkpoint as `model.py`. Built with
[VQLab](https://github.com/noahzelezny/VQLab).

MoE experts at d=4/K=512 (9-bit packed codes), with eight expert layers
promoted to d=4/K=2048 (11-bit) — layers 20, 27, 29, 31, 33, 34, 35, 39,
chosen by measured single-layer effect. Attention, embeddings and the output
head stay 8-bit affine; norms, routers and the full 347-tensor vision tower
stay bf16.

The affine builds compared against below are our own conversions of the same
base, made with the same tooling, scored on the same instrument.

Instead of rounding each weight onto a uniform grid the way affine
quantization does, VQ stores small groups of weights as indices into
codebooks fitted to the weights themselves — which is why it beats affine at
matched bytes below 6 bits. The method and results are in our paper,
[*Data-Free Vector Quantization Beats Affine Quantization at Matched Bytes
Below 6 Bits*](https://doi.org/10.5281/zenodo.22136000) (CC BY 4.0). This
build goes beyond the published recipe: it mixes codebook sizes, promoting
individual expert layers by measured effect (see **How it was built**).

## Requirements

The `glm5_next` architecture ships in **released `mlx-vlm` 0.6.17** — a
normal `pip install`, not a fork or an unmerged PR. The VQ runtime needs no
patches: it ships as `model.py` inside the checkpoint, declared via
`model_file` in `config.json`, and resolves under both `mlx-lm` and
`mlx_vlm`.

```bash
pip install mlx-vlm

python -m mlx_vlm generate \
  --model TheDrainFlorist/GLM-5.3-Flash-VQ-2.7bpw \
  --prompt "Explain vector quantization briefly." \
  --max-tokens 512
```

exo-ready: `config.json` carries `vision_config` and `image_token_id`, and
the vision tower ships bf16. For cluster serving use the
[`vq-serving`](https://github.com/noahzelezny/exo/tree/vq-serving) branch.

## Measured results

Referee: 2048 tokens; prose = WikiText, code = public mlx corpus (pinned
manifest), literary = Gutenberg. KL is against the bf16 teacher's cached
top-64 logits (captured mass 0.9906 on every row). This model's row was
measured on the published artifact itself, all three corpora, on the same
interpreter as every comparator row (mlx 0.32.2 / mlx-vlm 0.6.17).

| build | size | KL to bf16 (mnats/tok) | top-1 agreement | prose ppl | code ppl | literary ppl |
|---|---|---|---|---|---|---|
| VQ d4/K512 (this build's base) | 98.5 GiB | 348.82 | 84.0% | 2.5743 | 1.7107 | 1.6166 |
| **this model** | **101.9 GiB** | **293.84** | **85.7%** | **2.4014** | **1.6671** | **1.4811** |
| VQ d4/K2048 (uniform) | 116.3 GiB | 199.53 | 88.6% | 2.1954 | 1.6187 | 1.3402 |
| affine q3 (ours) | 129 GiB | 377.08 | 83.1% | 2.6824 | 1.7842 | 1.4731 |
| VQ d4/K8192 (uniform) | 134.0 GiB | 94.54 | 92.1% | 2.0379 | 1.5475 | 1.2154 |
| affine q4 (ours) | 166 GiB | 98.34 | 91.9% | 2.0263 | 1.5718 | 1.2025 |
| affine q6 (ours) | 239 GiB | 13.47 | 97.1% | 1.9285 | 1.4929 | 1.1660 |
| bf16 teacher | 598.5 GiB | 0 | 100% | 1.9024 | 1.4888 | 1.1580 |

**This is the fits-a-128 GB-Mac rung.** Against affine q3 it is 27 GiB
smaller *and* ~22% better on KL (293.84 vs 377.08), with clear wins on prose
and code; q3 edges it narrowly on literary (1.4731 vs 1.4811), the corpus
this family memorized hardest, and the gap closes with bits — the 116 GiB
rung wins literary outright. It is not q4-class: reaching q4 quality from
this family costs 134 GiB (the d4/K8192 rung, which matches or beats affine
q4 on every axis at 32 GiB less), and affine q4 itself is a 166 GiB
download. At this rung's memory budget it is the best thing we have
measured.

**Rank these by KL, not perplexity.** Perplexity is an aggregate over finite
text and absorbs offsetting errors; KL measures distance to the teacher's
distribution directly. KL here is prose-only (the teacher cache is a prose
cache).

**These perplexity scores aren't comparable across model families.** The
bf16 teacher has near-verbatim memorized the public corpora used here (mean
top-1 probability 0.857 on prose, measured), so absolute perplexity for this
family is contamination-dominated. KL to that teacher stays fully valid — a
sharp teacher is *harder* to track.

## Runtime

Measured on a Mac Studio M4 128 GB, single box, warm (cold first cycles page
badly at ~101 GiB resident; discard one before benchmarking):

| | |
|---|---|
| decode, stock generate | **20.3 tok/s** (flat out to 2000 tokens) |
| decode, MTP sidecar (`vqlab mtp-generate`) | 20.0 tok/s at 1000 tokens, 17.1 at 2000; acceptance ~0.75 |
| prefill | ~88 tok/s (lower bound, derived from wall time) |

## Speculative decoding (MTP) — optional sidecar

This repo includes `mtp-head-q6.safetensors` (6.09 GiB): GLM's own
multi-token-prediction head — `layers.45`, 889 tensors, 13.84 GiB as a bf16
graft, packed to q6. It is never named in the weight index, so stock loaders
ignore it entirely; it costs nothing unless you opt in by name.

**When enabled it adds ~6.3 GiB resident** (head weights + its cache).
Acceptance is **~0.75**, and the trunk verifies every drafted token by
exact rejection sampling, so the output distribution is exactly the base
model's. **On a single box it does not make decoding faster** — parity
with plain decode out to ~1000 generated tokens (20.0 vs 20.3 tok/s),
drifting ~15% behind by 2000. Where speculation pays is **pipelined
multi-node decoding** (exo clusters, for the rungs of this family too
large for one machine — see their cards): the [`mtp-stage1`](https://github.com/noahzelezny/exo/tree/mtp-stage1)
branch of our exo fork drafts with `EXO_MTP=1` set on every node, with
outputs exactly the base model's. Cluster throughput varies with shard
placement on mixed-generation hardware, so benchmark your own topology.
On one 128 GB box, run this model plain.

**Run the sidecar through `vqlab serve` (or `vqlab mtp-generate`), not
your own loop.** Verifying drafted tokens sends multi-token steps through
GLM's attention, and the upstream fast path only handles single tokens —
a 2-token verify falls off it and pays a measured 23x per-layer attention
tax that grows with context. Every vqlab drafting entry point installs
the fix (the fast path extended to short verify steps, numerically
equivalent to within one bf16 ULP per layer).

To use it:

```bash
git clone https://github.com/noahzelezny/VQLab && cd VQLab
python3 -m venv .venv && source .venv/bin/activate
pip install .
python -m vqlab.cli serve \
  --model TheDrainFlorist/GLM-5.3-Flash-VQ-2.7bpw \
  --sidecar mtp-head-q6.safetensors
```

OpenAI-compatible API on localhost; `vqlab mtp-generate` for one-shot CLI
use. Without `--sidecar`, nothing about the model changes.

The `mtp-head-q6.safetensors` files in our lineups (Qwen Flash 2.14 GiB,
Qwen 397B 5.41 GiB, GLM 6.09 GiB) are different heads with different
geometry and share only a filename — never cross-copy them between families.

## Memory, measured externally

Process RSS sampled at 5 Hz from outside the process, M4 128 GB, single box:

- **61.1 GiB** peak for load + 2048-token prefill + 128-token decode.
  Weights are memory-mapped, so RSS counts only pages actually touched —
  useful for comparing artifacts on the same instrument, **not for capacity
  planning**.
- **~100.9 GiB** sustained resident once routing has touched the experts;
  budget this. Trunk + MTP head is ~107 GiB — it runs on a 128 GB box and is
  tight there; close memory-heavy applications.

Long-prompt peaks stay near resident: the bundled runtime caps MLX's
buffer-reuse cache by default (`VQLAB_CACHE_LIMIT_GB` overrides, `0`
disables), measured free on wall time. Under exo, set
`EXO_MLX_CACHE_LIMIT_GB=6` and `EXO_MLX_MEM_LIMIT_GB` a few GiB under
physical RAM so an overrun is a traceback rather than a frozen Mac; a
26,423-token prompt through a 2-node pipeline held per-chunk transients
under 2 GiB, flat across all 13 chunks. If you run your own serving loop,
chunk the prefill (2048) and call `mx.eval([c.state for c in cache])` plus
`mx.clear_cache()` after every chunk — the chunk size only bounds the peak
if each chunk is actually forced.

## How it was built

Fitted data-free from the bf16 checkpoint — k-means / Lloyd over weight
subvectors, seed 1234, no Hessian, no activation statistics, no calibration
corpus. The base is a uniform d=4/K=512 fit of the 42 expert layers (L3–L44;
L45 is GLM's MTP layer and is never quantized), 98.55 GiB at 2.635 bpw.
Eight expert layers are then promoted to d=4/K=2048 (+0.42 GiB each), giving
the 101.93 GiB shipped size. Codes are packed sub-byte into uint32 words
(9 bits at K=512, 11 at K=2048) with an fp16 scale per (row, 64 weights).
The "2.7bpw" in the name is derived from the measured sizes; GiB is the
measured quantity and is what the card quotes.

**The eight promoted layers were chosen by measurement, not a heuristic**:
all 42 expert layers were promoted one at a time and scored on the
assembled model, and the eight with the largest measured effect shipped.
Targeting this way is 1.93x the efficiency of buying bits uniformly,
capturing 37% of the full K512→K2048 step's gain for 19% of its bytes.
Layer effects are base-specific and do not transfer to other rungs or
families; the full sweep, the controls, and the probe it falsified are in
the project ledger.

## Known limitations

- **This is the aggressive end of the ladder.** Prose perplexity is 1.26x
  the teacher's. If you have the memory, the 134 GiB rung reaches q4-class
  quality (not yet published).
- **q3 edges it on literary** (1.4731 vs 1.4811) — the one axis where
  smaller-and-better does not hold. VQ damage lands hardest on the most
  memorized corpus, so test a literary-heavy workload yourself.
- **Tight on 128 GB with the sidecar** (~107 GiB resident). Cold runs page;
  discard a cycle before benchmarking.
- **MTP is single-box parity, not a speedup** (19.99 vs 19.7 tok/s), and it
  must run through `vqlab serve` / `mtp-generate` — a homegrown verify loop
  makes it a net loss (see the MTP section).
- **Decode small-M kernel behaviour** is a known open frontier for this
  lineup, not a property of the quantization quality.

## Provenance and gates

Base model:
[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) —
**MIT licensed**; this is a quantized derivative and inherits that license.
Quantization: TheDrainFlorist, 2026. Release gates passed on this artifact:
file/index/tokenizer checks, bundle-runtime verbatim match, and a generation
smoke through the shipping runtime on Apple Silicon; the full three-corpus
referee row was measured against the published artifact directory itself.

The upstream authors ask that their technical report be cited in research
use:

```bibtex
@misc{glm5team2026glm5vibecodingagentic,
      title={GLM-5: from Vibe Coding to Agentic Engineering},
      author={GLM-5-Team and : and Aohan Zeng and Xin Lv and Zhenyu Hou and Zhengxiao Du and Qinkai Zheng and Bin Chen and Da Yin and Chendi Ge and Chenghua Huang and Chengxing Xie and Chenzheng Zhu and Congfeng Yin and Cunxiang Wang and Gengzheng Pan and Hao Zeng and Haoke Zhang and Haoran Wang and Huilong Chen and Jiajie Zhang and Jian Jiao and Jiaqi Guo and Jingsen Wang and Jingzhao Du and Jinzhu Wu and Kedong Wang and Lei Li and Lin Fan and Lucen Zhong and Mingdao Liu and Mingming Zhao and Pengfan Du and Qian Dong and Rui Lu and Shuang-Li and Shulin Cao and Song Liu and Ting Jiang and Xiaodong Chen and Xiaohan Zhang and Xuancheng Huang and Xuezhen Dong and Yabo Xu and Yao Wei and Yifan An and Yilin Niu and Yitong Zhu and Yuanhao Wen and Yukuo Cen and Yushi Bai and Zhongpei Qiao and Zihan Wang and Zikang Wang and Zilin Zhu and Ziqiang Liu and Zixuan Li and Bojie Wang and Bosi Wen and Can Huang and Changpeng Cai and Chao Yu and Chen Li and Chengwei Hu and Chenhui Zhang and Dan Zhang and Daoyan Lin and Dayong Yang and Di Wang and Ding Ai and Erle Zhu and Fangzhou Yi and Feiyu Chen and Guohong Wen and Hailong Sun and Haisha Zhao and Haiyi Hu and Hanchen Zhang and Hanrui Liu and Hanyu Zhang and Hao Peng and Hao Tai and Haobo Zhang and He Liu and Hongwei Wang and Hongxi Yan and Hongyu Ge and Huan Liu and Huanpeng Chu and Jia'ni Zhao and Jiachen Wang and Jiajing Zhao and Jiamin Ren and Jiapeng Wang and Jiaxin Zhang and Jiayi Gui and Jiayue Zhao and Jijie Li and Jing An and Jing Li and Jingwei Yuan and Jinhua Du and Jinxin Liu and Junkai Zhi and Junwen Duan and Kaiyue Zhou and Kangjian Wei and Ke Wang and Keyun Luo and Laiqiang Zhang and Leigang Sha and Liang Xu and Lindong Wu and Lintao Ding and Lu Chen and Minghao Li and Nianyi Lin and Pan Ta and Qiang Zou and Rongjun Song and Ruiqi Yang and Shangqing Tu and Shangtong Yang and Shaoxiang Wu and Shengyan Zhang and Shijie Li and Shuang Li and Shuyi Fan and Wei Qin and Wei Tian and Weining Zhang and Wenbo Yu and Wenjie Liang and Xiang Kuang and Xiangmeng Cheng and Xiangyang Li and Xiaoquan Yan and Xiaowei Hu and Xiaoying Ling and Xing Fan and Xingye Xia and Xinyuan Zhang and Xinze Zhang and Xirui Pan and Xu Zou and Xunkai Zhang and Yadi Liu and Yandong Wu and Yanfu Li and Yidong Wang and Yifan Zhu and Yijun Tan and Yilin Zhou and Yiming Pan and Ying Zhang and Yinpei Su and Yipeng Geng and Yong Yan and Yonglin Tan and Yuean Bi and Yuhan Shen and Yuhao Yang and Yujiang Li and Yunan Liu and Yunqing Wang and Yuntao Li and Yurong Wu and Yutao Zhang and Yuxi Duan and Yuxuan Zhang and Zezhen Liu and Zhengtao Jiang and Zhenhe Yan and Zheyu Zhang and Zhixiang Wei and Zhuo Chen and Zhuoer Feng and Zijun Yao and Ziwei Chai and Ziyuan Wang and Zuzhou Zhang and Bin Xu and Minlie Huang and Hongning Wang and Juanzi Li and Yuxiao Dong and Jie Tang},
      year={2026},
      eprint={2602.15763},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.15763},
}
```

Quantization method and measurement rules:
[**Data-Free Vector Quantization Beats Affine Quantization at Matched Bytes
Below 6 Bits**](https://doi.org/10.5281/zenodo.22136000) (CC BY 4.0).

Local artifact: `glm53_vq_packed_mix_best8`.
