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
# litbench: acc_norm is the metric of record — the hand-authored items carry
# a residual length bias (answers skew ~3 chars long), and byte-length
# normalization is what keeps that from scoring as literary understanding.
# See literary/README.md.
METRIC = {"hellaswag": "acc_norm", "piqa": "acc_norm", "winogrande": "acc",
          "litbench": "acc_norm"}

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


def load(name, directory):
    s = json.load(open(directory / f"{name}.samples.json"))
    out = {}
    for task, rows in s.items():
        m = METRIC[task]
        out[task] = {r["doc_id"]: float(r[m]) for r in rows}
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None,
                    help="results dir (default results_tasks/)")
    ap.add_argument("--tasks", default=None,
                    help="comma-separated subset of tasks to report")
    args = ap.parse_args()

    directory = pathlib.Path(args.dir) if args.dir else DIR
    tasks = ([t.strip() for t in args.tasks.split(",") if t.strip()]
             if args.tasks else list(METRIC))

    # Fall back to whatever samples files are present, so a new results dir
    # (results_literary/) works without editing ORDER. Keep ORDER's curated
    # labels and size-ascending sequence when the files match it.
    present = {p.name[:-len(".samples.json")]
               for p in directory.glob("*.samples.json")}
    order = [(n, l) for n, l in ORDER if n in present]
    order += [(n, n) for n in sorted(present - {n for n, _ in ORDER})]
    if not order:
        raise SystemExit(f"no *.samples.json in {directory}")

    data = {name: load(name, directory) for name, _ in order}
    label = dict(order)
    rng = random.Random(0)

    for a, b in itertools.combinations([n for n, _ in order], 2):
        print(f"\n=== {label[a]}  vs  {label[b]}")
        for task in tasks:
            if task not in data[a] or task not in data[b]:
                continue
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
