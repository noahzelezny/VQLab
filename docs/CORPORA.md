# Referee corpora — provenance and licensing

The scoring referee is only meaningful if everyone scores the same frozen
bytes, so the corpora ship in-repo. Never compare perplexity ACROSS corpora
— only models within one corpus.

| file | contents | license |
|---|---|---|
| `referee_corpus.txt` | WikiText-2 excerpt (Wikipedia-derived prose) | CC BY-SA (attribution: Wikipedia contributors; Merity et al., WikiText) |
| (code corpus) | **Not shipped.** The paper's code-ppl column was measured on a private corpus that is not redistributable. The prose and literary columns are fully reproducible from the shipped corpora; the code column is reproducible only in the statistical sense on a rebuilt corpus, which is a different instrument. Build one with `scripts/make_code_corpus.py` (records provenance + hashes in a manifest). Never put numbers from two corpora in one table. |
| `referee_corpus_literary.txt` | Excerpts from 10 public-domain works (Austen, et al.) via Project Gutenberg | Public domain (US); see `referee_corpus_literary.manifest.json` for the exact works, PG ids, and per-work character counts |

The literary manifest is the frozen definition of that corpus: work list,
Project Gutenberg ids, and byte contributions. If you rebuild the corpus,
you have a *different instrument* — name it as one.
