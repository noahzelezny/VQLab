# Community note — post to each 27B repo's Community tab (Noah posts)

Title: Bundle repair shipped — if `model.py` failed to load, re-download it

---

Heads up for everyone who downloaded this model: until today, the bundled
`model.py` could fail on a stock `mlx-lm` install with a
`ModuleNotFoundError`. The bug was on our side — the bundle depended on a
module that only existed in our internal environment, and our pre-release
testing ran in an environment that had it, so the failure was invisible
to us. The weights were always fine.

Fixed as of this revision: the bundle is now fully self-contained, and
our release gate now loads the model and generates tokens in a
verified-clean environment before anything uploads, so this class of
failure can't ship again. If you hit the error, just re-download
`model.py` (or `hf download <this-repo> model.py`) — no need to re-pull
the weights.

And a genuine ask: ~2k of you downloaded these and nobody reported the
breakage — please do! Open a discussion here for anything that doesn't
work, reads wrong in the card, or is slower than the numbers we publish.
Every number on the card comes from a measurement we can reproduce, and
we'd rather fix a real problem than not know about it. Feedback of any
kind is welcome — this line improves fastest when people push on it.
