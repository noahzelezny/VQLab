# LinkedIn post draft — VQLab release (2026-09-03)

For Noah's edit. Three lengths; pick one, tweak voice. Numbers are all
measured and defensible — every claim traces to a ledger entry. The
[bracketed] bits need your choice or a link.

---

## Short (the skimmable one)

Shipped: VQLab — vector-quantized builds of Qwen3.5-397B, Qwen3.8
Flash-Next, GLM-5.3-Flash, and more, running on Apple Silicon.

- 397B-class model on two Macs (RDMA over Thunderbolt 5, ~20 tok/s)
- 2.1–5.5 bits/weight via vector quantization — smaller than affine
  quants at matched-or-better quality
- Speculative decoding from each model's own MTP head: +34-43% decode,
  output distribution provably unchanged
- Custom Metal kernels for the VQ expert path (+8% end-to-end,
  verified bit-equivalent)

Every artifact ships with a runtime that loads on stock mlx-lm — no
fork required to run them. The serving fork, the release gates, and the
research ledgers (including the negative results) are public.

[link to HF collection] [link to repo]

## Medium (adds the engineering story)

Over the past weeks I built VQLab: a vector-quantization pipeline,
serving stack, and release process for running frontier-scale open
models on Apple Silicon — and today the refreshed lineup ships.

What's in it:

- **The models**: Qwen3.5-397B-A17B at 2.2–3.1 bpw, Qwen3.8 Flash-Next
  at 2.1–5.5 bpw, GLM-5.3-Flash at 2.7–3.6 bpw, plus 35B/27B lines.
  VQ codebooks instead of affine scales: at matched on-disk size, the
  VQ builds read ~5% fewer bytes per token and score better on
  perplexity referees.
- **A 397B model on two Macs**: pipeline/tensor sharding over
  Thunderbolt 5 with RDMA (+19% over TCP), ~20 tok/s decode.
- **Speculative decoding for free**: these architectures ship with a
  multi-token-prediction head that most runtimes throw away. VQLab
  packs it as an optional sidecar; the trunk verifies every draft, so
  the output distribution is exactly the base model's. Measured: +34%
  on an 80B (18.8 -> ~25 tok/s), +43% on GLM-5.3.
- **Kernel work with receipts**: five optimization arcs on the Metal
  expert kernels, +8% end-to-end. The wins shipped only after proving
  bit-identity (or, for one reduction, 1-ULP equivalence with measured
  zero quality delta — identical perplexity to 16 digits). The six
  refuted theories are in the ledger next to the two wins.
- **Release engineering**: after shipping one broken bundle early on,
  the upload path now refuses to publish anything that fails a
  load-and-generate gate in a clean environment. Tonight that gate
  caught (and fixed) four artifacts that would have broken on stock
  installs.

Everything loads on stock mlx-lm. [links]

## One-liner (if you'd rather comment-bait)

A 397B-parameter model, two Macs, one Thunderbolt cable, 20 tok/s.
Quantization is a systems problem, and the receipts are in the repo. [link]

---

Notes for you:
- The "+43% GLM" figure is tonight's 1.43x MTP measurement; the GLM
  sidecar ships in the same pass so the claim is live when posted.
- Avoid "fastest": spicyneuron's affine 2.6bit decodes faster on the
  cluster (29.9 vs 20.3). Our honest line is bytes/quality per byte,
  which the medium version states. If asked, own it — it reads well.
- Timing: post AFTER the push completes so the links resolve.
