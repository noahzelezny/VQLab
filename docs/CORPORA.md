# Referee corpora — provenance and licensing

The scoring referee is only meaningful if everyone scores the same frozen
bytes, so the corpora ship in-repo. Never compare perplexity ACROSS corpora
— only models within one corpus.

| file | contents | license |
|---|---|---|
| `referee_corpus.txt` | WikiText-2 excerpt (Wikipedia-derived prose) | CC BY-SA (attribution: Wikipedia contributors; Merity et al., WikiText) |
| `referee_corpus_code_public.txt` | **The canonical public code corpus** (since 2026-08-28): 6 files from mlx @ v0.30.0 (54f1cc6) — Python, Metal, C++ — 57,601 bytes; see its `.manifest.json` for per-file sha256. Built with `scripts/make_code_corpus.py`. All code-ppl numbers from the Flash-Next arc onward use THIS corpus. | Apache-2.0 (mlx, ml-explore) |
| (paper's code corpus) | **Not shipped, retired for new work.** The paper's code-ppl column was measured on a private corpus that is not redistributable; those numbers remain valid as relative comparisons among the paper's own builds. It is a DIFFERENT instrument from the public corpus above — never put numbers from the two in one table. |
| `referee_corpus_literary.txt` | Excerpts from 10 public-domain works (Austen, et al.) via Project Gutenberg | Public domain (US); see `referee_corpus_literary.manifest.json` for the exact works, PG ids, and per-work character counts |

The literary manifest is the frozen definition of that corpus: work list,
Project Gutenberg ids, and byte contributions. If you rebuild the corpus,
you have a *different instrument* — name it as one.
