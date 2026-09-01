# quantlab — Scout's quantization machine

Pinned, patched, isolated toolchain for producing Scout's quants. Exists so a stray
`pip upgrade` elsewhere can never silently erase the MoE fixes.

- `venv/` — mlx-optiq **pinned 0.4.18** (PPL scales are NOT comparable across optiq
  versions — 0.4.19 measures a different scale. Never mix versions in one comparison.)
- `patches/moe-allocator-fixes.patch` — the two env-gated MoE fixes applied to
  `optiq/core/optimizer.py` (already applied in this venv; re-apply after any
  deliberate upgrade with `patch -p0 < patches/...` against the new file, then
  re-verify with the dry-run below):
  - `OPTIQ_ATTN_FLOOR_BITS=4` — floor attention layers pre-greedy (per-layer KL is
    deceptively flat on MoE attention; unfloored, greedy inverts the proven recipe).
  - `OPTIQ_EXPERT_PARAM_MULT=<num_experts>` — batched-expert sweep entries count ONE
    expert's params while the bit choice applies to all N; without this the bit
    budget is fiction (a 35B checkpoint summed ~2B params).

Verify after any change (expects `achieved_bpw ≈ 2.604` on the 35B checkpoint):

```bash
OPTIQ_ATTN_FLOOR_BITS=4 OPTIQ_EXPERT_PARAM_MULT=256 ~/Documents/AgenicAI/quantlab/venv/bin/python -c "
import json
from optiq.core.sensitivity import SensitivityResult
from optiq.core import optimizer as opt
d=json.load(open('/Volumes/Thunderbay SSD/Exo Models/optiq-ab-35b/sensitivity_checkpoint.json'))
r=[SensitivityResult(layer_name=e['layer_name'],sensitivities={int(k):v for k,v in e['sensitivities'].items()},param_count=e['param_count']) for e in d]
print(opt.optimize_mixed_precision(r, target_bpw=2.6, candidate_bits=[2,3,4]).achieved_bpw)"
```

Full experiment history: research store `topic:llm-quantization`
(sources 1934a079 → f828e303 → e47cd33e → c91de858 → b293ff81 → 9fbc2733).

2026-08-09.


## Patch inventory (updated 2026-08-10, E13-E14)

`patches/quantlab-full.patch` is the CURRENT complete diff vs the pristine 0.4.18
wheel, covering BOTH files (supersedes moe-allocator-fixes.patch, which is
optimizer.py only):

- `core/optimizer.py` — forced-allocation modes (OPTIQ_FORCE_*), expert param mult
  (OPTIQ_EXPERT_PARAM_MULT), attention floor (OPTIQ_ATTN_FLOOR_BITS).
- `core/sensitivity.py` — bf16 sweep filter admits SwitchLinear (else routed experts
  are silently dropped: 391 vs 511 targets), and depth-prior modes for the static
  method: `OPTIQ_DEPTH_PRIOR=monotone|flat|reverse|reverse_experts` (+ OPTIQ_DEPTH_K,
  which only matters if it changes rank order — the allocator is rank-based, E13).

**Production 397B builds use `OPTIQ_DEPTH_PRIOR=reverse_experts`** (E14: builds the
bit-identical winner at every measured budget; strictly >= the stock U). Verify after
any reinstall: `OPTIQ_DEPTH_PRIOR=reverse_experts python -c "from optiq.core.sensitivity
import _structural_priority as sp; print(sp('model.layers.39.mlp.switch_mlp.up_proj',40))"`
must print 0.8 (not 0.9 — 0.9 means the patch is missing and you have the stock U).
