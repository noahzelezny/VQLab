#!/usr/bin/env python3
"""Thin wrapper: the resident referee ppl scorer lives in the package now
(src/vqlab/score/ppl_resident.py, `python -m vqlab.score.ppl_resident`).

    python scripts/score_ppl_resident.py --model DIR [--corpus F] [--max-tokens 2048]
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from vqlab.score.ppl_resident import main  # noqa: E402

if __name__ == "__main__":
    main()
