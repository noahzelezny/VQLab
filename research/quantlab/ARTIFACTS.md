# Artifact inventory

Generated from `/Volumes/Thunderbay SSD/Exo Models`. Sizes are `du` (block usage) and, where an
index exists, the exact sum of the shard bytes it names. **`du` is not a
size** -- an audit once mis-reported a 15.670 GiB artifact as 18.483 by
reading blocks. Cite `index_gib`.
**Total: 1,369.2 GiB across 14 artifacts.**


`rebuildable` says what deleting costs. See make_artifact_inventory.py
for what each value means; `vq-fit` is the only one that is not
reproducible, and per the 08-25 ruling that is not a veto on deletion
when the result is written down.

## base/teacher — 751.4 GiB, 1 artifacts

| artifact | du GiB | index GiB | tensors | vision | rebuildable |
|---|---|---|---|---|---|
| `Qwen--Qwen3.5-397B-A17B-bf16` | 751.4 | 751.3876 | 2924 | 333 | redownload |

## ours/published — 617.7 GiB, 13 artifacts

| artifact | du GiB | index GiB | tensors | vision | rebuildable |
|---|---|---|---|---|---|
| `TheDrainFlorist--Qwen3.5-397B-A17B-VQ-3bpw` | 143.7 | 143.6821 | 2545 | 333 | on-hf |
| `TheDrainFlorist--Qwen3.5-397B-A17B-VQ-2.6bpw` | 122.3 | 122.3051 | 2545 | 333 | on-hf |
| `TheDrainFlorist--Qwen3.5-397B-A17B-VQ-2.4bpw` | 111.6 | 111.6173 | 2545 | 333 | on-hf |
| `TheDrainFlorist--Qwen3.5-397B-A17B-VQ-2.2bpw` | 101.0 | 100.9712 | 2545 | 333 | on-hf |
| `TheDrainFlorist--Qwen3.6-35B-A3B-VQ-5.4bpw` | 22.2 | 22.2258 | 1810 | 333 | on-hf |
| `TheDrainFlorist--gemma-4-26b-a4b-it-VQ-6.2bpw` | 18.8 | 18.7354 | 1635 | 356 | on-hf |
| `TheDrainFlorist--Qwen3.6-35B-A3B-VQ-4.6bpw` | 18.7 | 18.7103 | 1810 | 333 | on-hf |
| `TheDrainFlorist--Qwen3.6-35B-A3B-VQ-3.8bpw` | 15.7 | 15.6702 | 1810 | 333 | on-hf |
| `TheDrainFlorist--Qwen3.8-27B-VQ-4.8bpw` | 15.5 | 15.4504 | 2180 | 333 | on-hf |
| `TheDrainFlorist--Qwen3.8-27B-VQ-4.5bpw` | 14.5 | 14.4541 | 2180 | 333 | on-hf |
| `TheDrainFlorist--Qwen3.6-35B-A3B-VQ-3.4bpw` | 13.8 | 13.7897 | 1810 | 333 | on-hf |
| `TheDrainFlorist--Qwen3.8-27B-VQ-3.9bpw` | 12.5 | 12.4676 | 2180 | 333 | on-hf |
| `TheDrainFlorist--gemma-4-e4b-it-VQ-PLE` | 7.4 | 7.3927 | 2768 | 661 | on-hf |
