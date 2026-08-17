#!/Users/noahzelezny/Documents/AgenicAI/quantlab/venv/bin/python
"""Paired analysis of the task-benchmark sweep.

Every model scored the SAME limit-1000 items (lm-eval seeded selection), so
model-vs-model comparison is PAIRED: item difficulty — the dominant variance
term — cancels. The independent stderr lm-eval prints (~±0.011-0.015 here)
is the right bar for absolute accuracy but the WRONG one for deltas; the
paired tests below are what can actually separate two quants.

Per pair and task:
  - delta = acc(A) - acc(B) over shared items
  - McNemar exact test on the discordant pairs (A-right/B-wrong vs
    A-wrong/B-right), two-sided binomial — the standard paired test for
    binary outcomes, no bootstrap noise
  - paired bootstrap 95% CI on the delta (10k resamples, fixed seed)

Reads results_tasks/*.samples.json (written by score_tasks_streaming.py
--output-dir). HellaSwag/PIQA use acc_norm (the headline metric); WinoGrande
has acc only.
"""
import itertools
import json
import math
import pathlib
import random

DIR = pathlib.Path(__file__).parent / "results_tasks"
METRIC = {"hellaswag": "acc_norm", "piqa": "acc_norm", "winogrande": "acc"}

ORDER = [  # ascending size, ours + comparators interleaved
    ("Qwen3.5-397B-A17B-VQ-2.2bpw", "VQ-2.2bpw (100.1G)"),
    ("Qwen3.5-397B-A17B-VQ-2.4bpw", "VQ-2.4bpw (110.8G)"),
    ("spicyneuron--Qwen3.5-397B-A17B-MLX-2.6bit", "spicy-2.6bit (120.6G)"),
    ("Qwen3.5-397B-A17B-VQ-3.1bpw", "VQ-3.1bpw (142.8G)"),
    ("spicyneuron--Qwen3.5-397B-A17B-MLX-3.5bit", "spicy-3.5bit (165.6G)"),
]


def binom_two_sided(k, n):
    """Exact two-sided binomial p under p=0.5 (sum of tails <= P(k))."""
    if n == 0:
        return 1.0
    pk = [math.comb(n, i) * 0.5 ** n for i in range(n + 1)]
    return min(1.0, sum(p for p in pk if p <= pk[k] + 1e-12))


def load(name):
    s = json.load(open(DIR / f"{name}.samples.json"))
    out = {}
    for task, rows in s.items():
        m = METRIC[task]
        out[task] = {r["doc_id"]: float(r[m]) for r in rows}
    return out


def main():
    data = {name: load(name) for name, _ in ORDER}
    label = dict(ORDER)
    rng = random.Random(0)

    for a, b in itertools.combinations([n for n, _ in ORDER], 2):
        print(f"\n=== {label[a]}  vs  {label[b]}")
        for task in METRIC:
            da, db = data[a][task], data[b][task]
            ids = sorted(set(da) & set(db))
            assert len(ids) == len(da) == len(db), (task, len(ids))
            xa = [da[i] for i in ids]
            xb = [db[i] for i in ids]
            n = len(ids)
            delta = (sum(xa) - sum(xb)) / n
            wins = sum(1 for u, v in zip(xa, xb) if u > v)
            losses = sum(1 for u, v in zip(xa, xb) if u < v)
            p = binom_two_sided(wins, wins + losses)
            deltas = []
            for _ in range(10000):
                idx = [rng.randrange(n) for _ in range(n)]
                deltas.append(sum(xa[i] - xb[i] for i in idx) / n)
            deltas.sort()
            lo, hi = deltas[249], deltas[9749]
            sig = "*" if p < 0.05 else " "
            print(f"  {task:11s} d={delta:+.4f}  disc {wins:3d}/{losses:3d} "
                  f" McNemar p={p:.4f}{sig}  boot95 [{lo:+.4f},{hi:+.4f}]")


if __name__ == "__main__":
    main()
