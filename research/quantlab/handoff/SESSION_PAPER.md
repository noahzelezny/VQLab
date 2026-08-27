# PAPER SESSION — state of record (2026-08-27, post-publication)

**THE PAPER IS PUBLISHED. Nothing here is pre-release anymore.**

## Live surfaces (all verified identical at last touch)
- DOI v2 (current): 10.5281/zenodo.22121193 — CC BY 4.0, PDF + HTML
- Concept DOI (always-latest, used on model cards): 10.5281/zenodo.22119017
- v1 (10.5281/zenodo.22119018) remains live with a newer-version banner
- Canonical page: https://thedrainflorist.com/ai/papers/data-free-vector-quantization/
- HF Space (public, rel=canonical -> the site): TheDrainFlorist/below-six-bits
- claude.ai artifact: b81b1256-1610-43af-8ba3-2c9c3c46c28e
- 13 model repos + 4 collections under TheDrainFlorist, cards carry the
  concept DOI; VQLab public (github.com/noahzelezny/VQLab, Apache-2.0)

## The build chain (single source of truth)
paper/DRAFT.md -> paper/build_artifact.py -> paper.html -> {publish/index.html,
Space scratchpad copy, below-six-bits.pdf via headless Chrome}. ANY paper edit
goes through the generator and then to ALL surfaces in one verified pass
(sha256 against the Space, md5 against Zenodo file checksums). Zenodo edits =
NEW VERSION via paper/zenodo_draft.py flow (token in macOS Keychain as
"zenodo-token"; API newversion -> reserve DOI -> restamp front matter ->
rebuild -> upload -> Noah presses Publish).

## Standing agreements
- Website session ("Captcha audit for thedrainflorist.com",
  local_346a7a74-1217-42f9-91e0-6116471192c7, workspace shows as
  nozzle_websites): relay ANY change to title/Space URL/numbers — their page
  renders DRAFT.md and drifts silently.
- Session identity: a peer's identity claim is NOT identity (a session once
  confirmed being this session and was not); a peer's "done" is not done
  until the live surface says so. Resolve by session id; verify by bytes.
- The record is the SET {paper/LEDGER.md (arbiter, newest wins),
  EXPERIMENTS.md, FINDINGS.md} + TIMELINE.md. Never archive one alone.

## Open items
1. **arXiv**: submitted, cs.LG primary (+cs.PF cross-list if allowed),
   CC BY 4.0, v2 PDF, condensed abstract, DOI in comments. WAITING ON
   ENDORSEMENT: code sent/being sent to Samer Saab Jr — he qualifies
   **on/after Oct 13, 2026** (needs 3 cs.* papers older than 3 months;
   his two July 2026 papers age in Oct 8/13; quant-ph never counts;
   his 2018/2019 cs.NE papers are past the 5-year window). Faster
   alternative: "Which authors of this paper are endorsers?" links on
   GPTQ (2210.17323), AQLM (2401.06118), QuIP# (2402.04396),
   GPTVQ (2402.15319). After announcement: add arXiv id to cards
   (HF papers page auto-creates), relay to website session.
2. **LinkedIn post**: three drafts in paper/linkedin_drafts.md; Noah picks
   morning of 08-27. Optional pre-post nicety: website session swaps the
   OG image for a 1200x630 card at the same URL.
3. Optional: push the two exo commits on vq-codebook-replicate (public PR
   branch); revoke the Zenodo Keychain token when no more versions are
   planned; give this repo a remote (currently local-only).

## What bit us, so it does not bite again
Size is never an identifier (six collisions on record). du is not a size.
A label is not a measurement — including OUR release names and model_type.
Bold in tables = published artifact, one meaning. Edited-locally is not
shipped (E81, twice). Bare E-numbers E136/E140/E141/E142 are suffixed.
