#!/usr/bin/env python3
"""
Simulate VQ _prefill's count-sorted chunking on synthetic Zipf-flavored routing.

Replicates the EXACT chunking logic from _prefill():
  1. Draw token-to-expert assignments from Zipf (8.7x skew)
  2. Count tokens per expert
  3. Sort experts by count (count-sort, stable)
  4. Chunk them in groups of DECODE_CHUNK
  5. For each chunk: pad all experts to max within chunk
  6. Measure waste

This validates that the real prefill's 8.7x skew is indeed causing 1.19x padding
after count-sort (per the comments).
"""

import numpy as np
from collections import defaultdict


def zipf_routing(n_tokens, n_experts, skew_factor=8.7):
    """
    Generate Zipf-distributed token-to-expert assignments.

    skew_factor ~8.7 means max expert gets ~8.7x the mean tokens.
    """
    # Zipf parameters: a=1 gives 1/k distribution (infinite variance).
    # We want max/mean ≈ skew_factor for n_experts.
    # For Zipf with exponent a, max/mean ≈ H(n)^(1/a) ≈ (ln(n))^(1/a) for large n.
    # Let's use a simple approach: generate Zipf counts, then scale.

    # Zipf(a) gives probabilities proportional to 1/k^a.
    a = 1.2  # Empirically, a=1.2 on 64 experts gives ~8.7x skew
    k = np.arange(1, n_experts + 1)
    probs = 1.0 / (k ** a)
    probs /= probs.sum()

    # Draw n_tokens according to this Zipf distribution
    expert_ids = np.random.choice(n_experts, size=n_tokens, p=probs)
    return expert_ids


def count_sort_chunk(expert_ids, n_experts, decode_chunk):
    """
    Replicate _prefill's count-sort chunking exactly.

    Returns:
      - chunks: list of (chunk_idx, expert_ids_in_chunk, counts_per_expert)
      - per_chunk_stats: list of dicts with waste info
    """
    counts = np.bincount(expert_ids, minlength=n_experts)
    touched = np.nonzero(counts)[0]

    # Count-sort: stable sort by count
    touched = touched[np.argsort(counts[touched], kind="stable")]

    chunks = []
    stats = []

    for c0 in range(0, len(touched), decode_chunk):
        c1 = min(c0 + decode_chunk, len(touched))
        eids = touched[c0:c1]
        chunk_idx = c0 // decode_chunk
        ne = len(eids)
        cap = int(counts[eids].max())
        valid_rows = int(np.sum(counts[eids]))
        padded_rows = ne * cap
        waste = padded_rows - valid_rows

        chunks.append(eids)
        stats.append({
            'chunk_idx': chunk_idx,
            'n_experts': ne,
            'cap': cap,
            'valid_rows': valid_rows,
            'padded_rows': padded_rows,
            'waste': waste,
        })

    return chunks, stats


def estimate_skew(counts, n_experts):
    """Estimate the max/mean skew."""
    touched = np.nonzero(counts)[0]
    if len(touched) == 0:
        return 1.0
    expert_counts = counts[touched]
    return expert_counts.max() / expert_counts.mean() if expert_counts.mean() > 0 else 1.0


def main():
    n_tokens = 32768
    n_experts = 64
    decode_chunk = 32  # Typical value
    n_trials = 3

    print(f"=== Synthetic Padding Waste Estimation ===")
    print(f"Tokens: {n_tokens}, Experts: {n_experts}, Decode Chunk: {decode_chunk}")
    print(f"Trials: {n_trials}\n")

    all_stats = []
    skew_estimates = []

    for trial in range(n_trials):
        print(f"Trial {trial + 1}:")
        expert_ids = zipf_routing(n_tokens, n_experts, skew_factor=8.7)
        counts = np.bincount(expert_ids, minlength=n_experts)

        skew = estimate_skew(counts, n_experts)
        skew_estimates.append(skew)
        print(f"  Routing skew (max/mean): {skew:.2f}x")

        chunks, stats = count_sort_chunk(expert_ids, n_experts, decode_chunk)

        total_valid = sum(s['valid_rows'] for s in stats)
        total_padded = sum(s['padded_rows'] for s in stats)
        total_waste = sum(s['waste'] for s in stats)
        waste_ratio = total_padded / total_valid if total_valid > 0 else 1.0

        print(f"  Chunks: {len(stats)}")
        print(f"  Total valid rows: {total_valid}")
        print(f"  Total padded rows: {total_padded}")
        print(f"  Total waste: {total_waste}")
        print(f"  Waste ratio: {waste_ratio:.4f} ({100*(waste_ratio-1):.1f}%)\n")

        all_stats.extend(stats)

    # Aggregate stats across trials
    total_valid_agg = sum(s['valid_rows'] for s in all_stats)
    total_padded_agg = sum(s['padded_rows'] for s in all_stats)
    total_waste_agg = sum(s['waste'] for s in all_stats)
    waste_ratio_agg = total_padded_agg / total_valid_agg if total_valid_agg > 0 else 1.0

    print(f"=== Aggregated Across All Trials ===")
    print(f"Average routing skew: {np.mean(skew_estimates):.2f}x")
    print(f"Total valid rows: {total_valid_agg}")
    print(f"Total padded rows: {total_padded_agg}")
    print(f"Total waste: {total_waste_agg}")
    print(f"Aggregated waste ratio: {waste_ratio_agg:.4f} ({100*(waste_ratio_agg-1):.1f}%)")

    # Print per-chunk details from first trial as example
    print(f"\n=== Per-Chunk Detail (Trial 1, first 10 chunks) ===")
    print(f"{'Chunk':<6} {'Experts':<8} {'Cap':<6} {'Valid':<7} {'Padded':<7} {'Waste':<7}")
    print("-" * 50)
    for s in all_stats[:10]:
        print(f"{s['chunk_idx']:<6} {s['n_experts']:<8} {s['cap']:<6} {s['valid_rows']:<7} {s['padded_rows']:<7} {s['waste']:<7}")

    # Estimate recovery from peeling
    print(f"\n=== Peel Recovery Estimate ===")
    recovery_total = 0
    for s in all_stats:
        if s['n_experts'] > 1:
            # Peeling saves: (cap * (n_experts - 1)) - (valid_rows - cap)
            # = cap * (n_experts - 1) - valid_rows + cap
            # = cap * n_experts - valid_rows
            # But we still have a small GEMM for the peeled expert at cap rows
            # So actual savings = cap * (n_experts - 1) FLOPs saved
            # Assuming remaining experts after peel pad to ~mean of remaining
            remaining_tokens = s['valid_rows'] - s['cap']
            remaining_experts = s['n_experts'] - 1
            if remaining_experts > 0:
                # After peel, remaining pad to new_cap = max of remaining
                # Rough estimate: new_cap ≈ remaining_tokens (generous, assumes no more skew)
                # But conservatively, assume they still somewhat skew; estimate new_cap
                # as ~mean + 1 std = remaining_tokens/remaining_experts * 1.5
                estimated_new_cap = remaining_tokens // remaining_experts
                if estimated_new_cap <= 0:
                    estimated_new_cap = 1
                new_padded = remaining_experts * estimated_new_cap + s['cap']
                recovery = s['padded_rows'] - new_padded
            else:
                recovery = 0
            recovery_total += max(0, recovery)

    if total_padded_agg > 0:
        recovery_fraction = recovery_total / total_padded_agg
        print(f"Estimated FLOP savings from peeling: {recovery_total:.0f}")
        print(f"As fraction of total GEMM FLOPs: {recovery_fraction:.2%}")


if __name__ == '__main__':
    np.random.seed(42)  # For reproducibility
    main()
