# registry/ — every artifact, from facts its bytes prove

`artifacts.jsonl`: one line per artifact instance (host + path), written by
`python src/vqlab/records/registry.py scan <root>` (a `vqlab registry` CLI
command once the 2026-09-26 night-4 freeze on cli.py lifts). Fields:
runtime `model.py` md5 + profile + VQ_* flags, geometry mix, text GiB,
shard fingerprint (sizes + head hashes), build-record id if any. Nothing is
transcribed from notes.

    python src/vqlab/records/registry.py list --grep TheDrainFlorist
    python src/vqlab/records/registry.py hub --all      # local vs Hub, metadata only

## First Hub drift run (2026-09-26, Exo Models vs HF, LFS by size)

- **Flash-Next 2.1 / 3.2 / 4.4: local serving copies are DIFFERENT WEIGHTS
  from the published repos** (3-4 shards + config.json differ in size). They
  are byte-identical to lab candidates qwen4exp_vq_packed_mixL01 / _31mix6 /
  _92mix6. Anything scored or served from Exo Models for these rungs is not
  the published artifact.
- GLM-5.3 2.7 / 3.1 / 3.6: model.py differs from the Hub (local v1.5 / v2 / v2).
- gemma-4 26B and e4b: the Hub repos carry a stray
  `__pycache__/model.cpython-312.pyc` (published by accident).
- 397B 2.2 / 2.4 / 2.6 / 3.1: local == Hub (besides local backup files).
- 35B, 27B, Flash 5.5: README (and the 35B ladder chart) only.

Profile spread (from the registry): 35B 3.4 = v1.5 while 3.8/4.6/5.4 = v2;
397B bundles 151f1eda (2.2, shared with 35B 3.4) and ca3645e0 (2.4/2.6/3.1);
27B = 545d4810 on all three; Flash 2.1 = v2, 3.2/4.4/5.5 = v1.5.
