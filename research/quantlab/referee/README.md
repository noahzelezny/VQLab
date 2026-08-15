# Referee — score any model on the one true corpus, no Scout/Claude needed

Everything here runs on python3 stdlib against a live exo cluster. The corpus
(`referee_corpus.txt`, sha256 `81ea5b79…`) is the SAME 60k-char wikitext slice
behind every E17 number, so results are directly comparable:

| model | PPL |
|---|---|
| t2.1-revexp | **9.106** |
| spicyneuron 2.6bit | 13.026 |
| t2.4-revexp | 18.948 |
| t2.6-revexp | 23.980 |

## The 4-bit max-RAM session (209G — needs everything else closed)

1. Quit Scout server, Claude, browsers, everything, on BOTH boxes.
2. Raise wired limits (sudo, resets at reboot):
   - M3: `sudo sysctl iogpu.wired_limit_mb=92160` (90 GiB)
   - M4: `sudo sysctl iogpu.wired_limit_mb=122880` (120 GiB)
3. Start exo on both boxes; wait ~30 s after the second node joins.
4. Inspect placements first:
   `python3 score_via_exo.py mlx-community/Qwen3.5-397B-A17B-4bit --list`
5. Score (caps keep it from picking a placement that will OOM):
   `python3 score_via_exo.py mlx-community/Qwen3.5-397B-A17B-4bit --cap "Noah's Mac Studio=88,NozzleBook Pro=118"`

It waits through the load (up to 40 min), prints the PPL vs the 9.106
baseline, and evicts the instance when done (`--keep` to leave it up).

Note: a 209G model split 2 ways is ~104 G/side — MORE than the M3 can wire
even at 90 G. Straight Tensor may not fit; if no Tensor preview fits the
caps, check `--list` for an uneven Pipeline placement and relax the script's
Tensor-only pick by using `--runners 2` output to decide by hand. Borderline
by design; scoring it at all is the win.

`score_via_exo.py <model> --runners 1` also works for single-node models.
`referee.py <model>` is the bare scorer (no placement) if an instance is
already up.
