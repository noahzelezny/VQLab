# Referee corpora — provenance and licensing

The scoring referee is only meaningful if everyone scores the same frozen
bytes, so the corpora ship in-repo. Never compare perplexity ACROSS corpora
— only models within one corpus.

| file | contents | license |
|---|---|---|
| `referee_corpus.txt` | WikiText-2 excerpt (Wikipedia-derived prose) | CC BY-SA (attribution: Wikipedia contributors; Merity et al., WikiText) |
| `referee_corpus_code.txt` | **PROVENANCE UNRESOLVED — do not redistribute.** 12 files: 8 from `exo` (Apache-2.0, attribution not yet attached) and 4 from a private application. An earlier revision of this table said "mlx-lm (MIT)" — wrong, concluded from an import statement in the first file. Fate pending owner decision; see the repo issue tracker before relying on this column's reproducibility. |
| `referee_corpus_literary.txt` | Excerpts from 10 public-domain works (Austen, et al.) via Project Gutenberg | Public domain (US); see `referee_corpus_literary.manifest.json` for the exact works, PG ids, and per-work character counts |

The literary manifest is the frozen definition of that corpus: work list,
Project Gutenberg ids, and byte contributions. If you rebuild the corpus,
you have a *different instrument* — name it as one.
