#!/usr/bin/env python3
"""Parse VQ_PREFILL_STATS output and report padding waste + recovery potential."""

import sys
import re
from collections import defaultdict


def parse_log_file(logfile):
    """Extract all VQ_PREFILL_CHUNK and VQ_PREFILL_STATS lines."""
    chunks = []
    summary = None

    with open(logfile, 'r') as f:
        for line in f:
            line = line.rstrip()

            # Match VQ_PREFILL_CHUNK lines
            m = re.search(
                r'VQ_PREFILL_CHUNK chunk_idx=(\d+) n_experts=(\d+) cap=(\d+) valid_rows=(\d+) padded_rows=(\d+)',
                line
            )
            if m:
                chunk_idx = int(m.group(1))
                n_experts = int(m.group(2))
                cap = int(m.group(3))
                valid_rows = int(m.group(4))
                padded_rows = int(m.group(5))
                chunks.append({
                    'chunk_idx': chunk_idx,
                    'n_experts': n_experts,
                    'cap': cap,
                    'valid_rows': valid_rows,
                    'padded_rows': padded_rows,
                })
                continue

            # Match VQ_PREFILL_STATS summary line
            m = re.search(
                r'VQ_PREFILL_STATS total_valid=(\d+) total_padded=(\d+) waste_ratio=([\d.]+)',
                line
            )
            if m:
                summary = {
                    'total_valid': int(m.group(1)),
                    'total_padded': int(m.group(2)),
                    'waste_ratio': float(m.group(3)),
                }
                continue

    return chunks, summary


def compute_peel_recovery(chunks):
    """
    For each chunk, compute the potential recovery from peeling the heaviest
    expert into a solo GEMM (removes it from padding).

    Recovery = (original_padded - new_padded) / total_padded_across_all_chunks
    """
    if not chunks:
        return None, None, None

    # Compute routing distribution per chunk
    chunk_expert_counts = []
    for chunk in chunks:
        # We need the per-expert counts, but we only have aggregates.
        # Approximate: assume experts in a chunk are SIMILAR size (they are sorted by count)
        # so the max and min are not vastly different. For a rough estimate,
        # assume a uniform distribution with one expert at max.
        mean_count = chunk['valid_rows'] / chunk['n_experts']
        max_count = (chunk['cap'])  # This is actually the MAX expert count, not a per-expert thing
        chunk_expert_counts.append({
            'chunk_idx': chunk['chunk_idx'],
            'mean_count': mean_count,
            'max_expert_rows': chunk['cap'],  # Max rows seen in this chunk
            'n_experts': chunk['n_experts'],
            'padded': chunk['padded_rows'],
        })

    # For each chunk, compute: if we peel the max expert into solo GEMM,
    # the remaining (n_experts-1) experts still pad to (second_max), and the max expert
    # gets its own exact GEMM (no padding).
    total_padded = sum(c['padded_rows'] for c in chunks)
    recovery_flops = 0

    for c in chunk_expert_counts:
        # Original: ne experts padded to cap each = ne * cap FLOPs
        # After peeling: (ne-1) experts padded to (second_max) each, + 1 expert at max_rows
        # Approximate: assume after removing max, the remaining experts pad to ~(mean_count).
        # So recovery = (cap - mean_count) * (we save on one expert)

        if c['n_experts'] > 1:
            # Simplified model: peeling the max saves us one expert's padding
            # (assuming the next-max is much smaller).
            # Actual formula: new_padded = (ne-1)*second_max + max_rows
            # Conservative estimate: second_max ≈ mean of non-max
            # But we don't have per-expert data, so approximate:
            # Total valid rows = max + (others)
            # Others padded to max previously, now pad to ~(total_valid - max) / (ne-1)
            # Recovery = max * (n_experts - 1) - (total_valid - max)
            # Simplified: recovery = (max * ne - total_valid) - (max * (ne-1) - (total_valid - max))
            #           = max*ne - total_valid - max*(ne-1) + total_valid - max
            #           = max - max*(ne-1) + max*(ne-1)
            # Actually, let me simplify even more:
            # Savings = we no longer pad (ne-1) experts to cap, only to second_max
            # Approximation: savings ≈ (cap - mean) * (ne - 1)
            approx_second_max = (c['mean_count'] * c['n_experts'] - c['max_expert_rows']) / (c['n_experts'] - 1) if c['n_experts'] > 1 else 0
            if approx_second_max < 0:
                approx_second_max = c['mean_count']
            recovery = c['max_expert_rows'] * c['n_experts'] - c['max_expert_rows'] - (c['n_experts'] - 1) * approx_second_max
        else:
            recovery = 0

        recovery_flops += max(0, recovery)

    projected_recovery_fraction = recovery_flops / total_padded if total_padded > 0 else 0
    return recovery_flops, projected_recovery_fraction, chunk_expert_counts


def main():
    if len(sys.argv) < 2:
        print("Usage: harvest_prefill_stats.py <logfile>")
        sys.exit(1)

    logfile = sys.argv[1]
    chunks, summary = parse_log_file(logfile)

    if not chunks:
        print("No VQ_PREFILL_CHUNK lines found in log file.")
        sys.exit(1)

    print("\n=== Per-Chunk Padding Waste ===")
    print(f"{'Chunk':<6} {'Experts':<8} {'Cap':<6} {'Valid':<7} {'Padded':<7} {'Waste':<7} {'Waste%':<8}")
    print("-" * 60)

    for chunk in chunks:
        waste = chunk['padded_rows'] - chunk['valid_rows']
        waste_pct = 100.0 * waste / chunk['padded_rows'] if chunk['padded_rows'] > 0 else 0
        print(f"{chunk['chunk_idx']:<6} {chunk['n_experts']:<8} {chunk['cap']:<6} {chunk['valid_rows']:<7} {chunk['padded_rows']:<7} {waste:<7} {waste_pct:<8.1f}")

    print()
    if summary:
        print(f"=== Total Stats ===")
        print(f"Total Valid Rows:  {summary['total_valid']}")
        print(f"Total Padded Rows: {summary['total_padded']}")
        print(f"Total Waste Ratio: {summary['waste_ratio']:.4f}")
        print(f"Total Waste %:     {100 * (summary['waste_ratio'] - 1):.1f}%")
    else:
        total_valid = sum(c['valid_rows'] for c in chunks)
        total_padded = sum(c['padded_rows'] for c in chunks)
        waste_ratio = total_padded / total_valid if total_valid > 0 else 0
        print(f"=== Aggregated Stats (from chunks) ===")
        print(f"Total Valid Rows:  {total_valid}")
        print(f"Total Padded Rows: {total_padded}")
        print(f"Total Waste Ratio: {waste_ratio:.4f}")
        print(f"Total Waste %:     {100 * (waste_ratio - 1):.1f}%")

    # Estimate recovery from peeling
    print()
    recovery_flops, recovery_frac, chunk_data = compute_peel_recovery(chunks)
    if recovery_flops is not None:
        print(f"=== Projected Recovery from Peeling Max Expert per Chunk ===")
        print(f"Estimated FLOP Recovery: {recovery_flops:.0f}")
        print(f"As Fraction of Total GEMM FLOPs: {recovery_frac:.2%}")


if __name__ == '__main__':
    main()
