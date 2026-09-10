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
- multimodal
---

# GLM-5.3-Flash-VQ-3.1bpw

**116.3 GiB — the balanced GLM rung, for exo clusters and 192 GB-class
Macs.**

A data-free vector-quantized build of
[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) for Apple
Silicon, fitted from the bf16 checkpoint (598.5 GiB) with no calibration
corpus. Built with [VQLab](https://github.com/noahzelezny/VQLab).

MoE experts uniformly at **d=4 / K=2048** (11-bit packed codes). Attention,
embeddings and the output head stay at 8-bit affine; norms, routers and the
full 347-tensor vision tower stay bf16. GLM's MTP layer (`layers.45`) is
never quantized.

Instead of rounding each weight onto a uniform grid the way affine
quantization does, VQ stores small groups of weights as indices into
codebooks fitted to the weights themselves — which is why it beats affine at
matched bytes below 6 bits. The method and results are in our paper,
[*Data-Free Vector Quantization Beats Affine Quantization at Matched Bytes
Below 6 Bits*](https://doi.org/10.5281/zenodo.22119017) (CC BY 4.0).

The affine builds compared against below are our own conversions of the same
base, made with the same tooling and scored on the same instrument.

## Requirements

The `glm5_next` architecture ships in **released `mlx-vlm` 0.6.17** — a
normal `pip install`, not a fork. The VQ runtime needs no patches: it ships
inside the checkpoint as `model.py`, declared via `model_file` in
`config.json`, and resolves under both `mlx-lm` and `mlx_vlm`.

```bash
pip install 'mlx-vlm>=0.6.17'
```

exo-ready: `config.json` carries `vision_config` and `image_token_id`, and
the vision tower ships bf16. For cluster serving use the
[`vq-serving`](https://github.com/noahzelezny/exo/tree/vq-serving) branch.

## Changelog

### 2026-09-09 — runtime refresh

**Runtime refresh.** The bundled `model.py` is updated so downloaders run
exactly the code that was benchmarked; dense bundles now carry both runtimes.

- Faster prefill on affected geometries, measured per rung: a
  device-codebook kernel arm for large-codebook geometries (up to 1.46x on
  affected rungs), a ragged-subvector relaxation (up to 1.34x on affected
  rungs), a fused d8 arm, and a routing memo (≈2%). No blanket speedup is
  claimed across the lineup — gains apply only where the geometry engages
  the new paths.
- Output quality is unchanged: the kernel changes are bit-identical or
  1-ULP-equivalent, and the routing memo is bit-identical (logits checksum
  verified).
- Speculative decoding (MTP): on repos that ship
  `mtp-head-q6.safetensors`, the sidecar works with the exo fork branch
  `mtp-stage1` (github.com/noahzelezny/exo) — launch each node with
  `exo --mtp` (or set `EXO_MTP=1`). Note: with MTP enabled, exo serves
  requests sequentially (the batch engine has no MTP path), so leave it
  off for concurrent / multi-agent workloads.

## Measured results

Referee: 2048 tokens; prose = WikiText, code = public mlx corpus (pinned
manifest), literary = Gutenberg. KL is against the bf16 teacher's cached
top-64 logits (captured mass 0.9906 on every row).

| build | size | KL to bf16 (mnats/tok) | top-1 agreement | prose ppl | code ppl | literary ppl |
|---|---|---|---|---|---|---|
| VQ 2.7bpw (mixed best-8) | 101.9 GiB | 293.84 | 85.7% | 2.4014 | 1.6671 | 1.4811 |
| **this model (d4/K2048)** | **116.3 GiB** | **199.53** | **88.6%** | **2.1954** | **1.6187** | **1.3402** |
| affine q3 (ours) | 129 GiB | 377.08 | 83.1% | 2.6824 | 1.7842 | 1.4731 |
| VQ 3.6bpw (d4/K8192) | 134.0 GiB | 94.54 | 92.1% | 2.0379 | 1.5475 | 1.2154 |
| affine q4 (ours) | 166 GiB | 98.34 | 91.9% | 2.0263 | 1.5718 | 1.2025 |
| affine q6 (ours) | 239 GiB | 13.47 | 97.1% | 1.9285 | 1.4929 | 1.1660 |
| bf16 teacher | 598.5 GiB | 0 | 100% | 1.9024 | 1.4888 | 1.1580 |

**Against affine at matched bytes:** this build is **13 GiB smaller than
affine q3 and 47% better on KL** (199.53 vs 377.08). It sits well short of
q4-class quality — if that is what you need, the
[3.6bpw rung](https://huggingface.co/TheDrainFlorist/GLM-5.3-Flash-VQ-3.6bpw)
matches or beats affine q4 on every axis at 32 GiB less.

**Rank these by KL, not perplexity.** Perplexity is an aggregate over finite
text and absorbs offsetting errors; KL measures distance to the teacher's
distribution directly.

**These perplexity scores aren't comparable across model families.** The
bf16 teacher has near-verbatim memorized the public corpora used here (mean
top-1 probability 0.857 on prose, measured), so absolute perplexity for this
family is contamination-dominated. KL to that teacher stays fully valid — a
sharp teacher is *harder* to track.

## Runtime

This rung exceeds any single machine we gate on (116 GiB of weights against
a 128 GB box), so it was verified on a **2-node exo pipeline** (M3 Ultra
96 GB + M4 128 GB, Thunderbolt): the release gate's cluster smoke
generated coherent tokens through this exact artifact, with the peer
rank's copy identity-checked (runtime/config/index hashes and every shard
size) before the generation counted. Single-box throughput numbers do not
exist for this rung and none are quoted; the bundled runtime is
byte-identical to the one gated and generation-verified single-box on the
[2.7bpw rung](https://huggingface.co/TheDrainFlorist/GLM-5.3-Flash-VQ-2.7bpw).

## Memory

- Download: **116.3 GiB** (19 shards); repo total including the sidecar is
  ≈122.4 GiB. Expect resident memory near the download size once routing has
  touched all experts; a single 128 GB Mac is not enough headroom — this is
  a 2-node exo or 192 GB-class rung.
- **Keep prefill bounded.** The bundled runtime caps MLX's buffer-reuse
  cache (4 GiB default, `VQLAB_CACHE_LIMIT_GB` to override, `=0` disables),
  which is what keeps long-prompt peaks near resident size. Under exo,
  additionally set `EXO_MLX_CACHE_LIMIT_GB=6` and `EXO_MLX_MEM_LIMIT_GB` a
  few GiB under physical RAM so an overrun is a traceback, not a frozen Mac.
- If you run your own serving loop: chunk the prefill (2048) and call
  `mx.eval([c.state for c in cache])` plus `mx.clear_cache()` after
  **every** chunk — the chunk size only bounds the peak if each chunk is
  actually forced.
- Single-box peak memory is not measured for this rung: it does not
  fit one machine. Cluster figures are above.

## Run it

```bash
python -m mlx_vlm generate \
  --model TheDrainFlorist/GLM-5.3-Flash-VQ-3.1bpw \
  --prompt "Explain vector quantization briefly." \
  --max-tokens 512
```

The VQ runtime needs no patches — it ships inside the checkpoint as
`model.py`, declared via `model_file` in `config.json`, and resolves under
both `mlx-lm` and `mlx_vlm`.

## Speculative decoding (MTP)

`mtp-head-q6.safetensors` (6.09 GiB) is GLM's own multi-token-prediction
head — `layers.45`, 889 tensors, packed to q6. The head is shared across the
GLM-5.3-Flash-VQ lineup because that layer is never quantized in any rung;
it was validated end-to-end on the 2.7bpw rung (acceptance **0.827** over
greedy decoding, output distribution exactly the base model's via rejection
sampling). It is never named in the weight index, so stock loaders ignore it
entirely; nothing changes unless you opt in.

**Cluster speculative decoding is live for this rung** on the [`mtp-stage1`](https://github.com/noahzelezny/exo/tree/mtp-stage1)
branch of our exo fork: set `EXO_MTP=1` on every node and serve as usual —
drafting engages automatically, and the trunk verifies every drafted token
by exact rejection sampling, so outputs are exactly the base model's.
Validated on this exact artifact on a 2-node pipeline; acceptance 0.64–0.77
depending on content. **Throughput varies with shard placement** on
mixed-generation clusters (chip generations trade compute against
bandwidth), so benchmark your own topology. The head adds ≈6.3 GiB on the
rank that draws it.

On a single box the MTP head roughly breaks even against plain decode
(measured on the 2.7bpw rung: 19.99 vs 19.7 tok/s); its payoff is pipelined
serving, where drafting hides inter-stage latency.

> The `mtp-head-q6.safetensors` files in our lineups (Qwen Flash 2.14 GiB,
> Qwen 397B 5.41 GiB, GLM 6.09 GiB) are **different heads with different
> geometry** across families and share only a filename. Never cross-copy
> them between families.

## Methodology

Fitted **data-free** from the bf16 checkpoint — k-means / Lloyd over weight
subvectors, seed 1234, no Hessian, no activation statistics, no calibration
corpus. All 42 expert layers (L3–L44) at uniform d=4/K=2048; codes are
packed sub-byte into uint32 words (11 bits per code) with an fp16 scale per
(row, 64 weights). The seed-noise floor for this family's geometry is
6.32 mnats on KL, well below every gap in the table above.

## Verification

Release gates passed on this artifact before upload: file, index and
tokenizer checks, a verbatim match between the bundled runtime and its
source, and a generation smoke through the shipping runtime on Apple
Silicon. The upload path runs the gate itself and refuses to publish
without it.
## Limitations

- **Verified on a 2-node cluster, not single-box.** Generation on this
  exact artifact ran through an exo pipeline; no single-box throughput or
  peak-memory figures exist yet.
- **Perplexity scores aren't comparable across model families** — see the
  note under Measured results.
- MTP throughput on clusters is placement-sensitive (see the MTP
  section); single-box, this rung does not fit one machine.

## Paper

The method, the full model ladder, the negative results, and the measurement
rules behind every number here:
[**Data-Free Vector Quantization Beats Affine Quantization at Matched Bytes
Below 6 Bits**](https://doi.org/10.5281/zenodo.22119017) (CC BY 4.0) ·
code: [VQLab](https://github.com/noahzelezny/VQLab) ·
web version: [Space](https://huggingface.co/spaces/TheDrainFlorist/below-six-bits)

## Provenance

Base model: [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) — **MIT licensed**.
This is a quantized derivative and inherits that licence; using it
means accepting the base model's terms.
Quantization: TheDrainFlorist, 2026.

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

Built with MLX and [VQLab](https://github.com/noahzelezny/VQLab).
