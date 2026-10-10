# util/ — Small helpers every stage shares

## Inputs
Paths, bytes, strings. Nothing model-shaped.

## Process
`hashing.py` (`sha256_file` with head/algo/missing_ok, `sha256_bytes`, `git_blob_sha1`, `fingerprint(dir)`: an input directory's cheap identity), `clock.py` (`now()` aware UTC datetime, `now_iso()`, `slug()`), `log.py` (`log()`, `make_log(prefix, stamp)`: flushed progress lines), `fs.py` (`dir_bytes()`: regular-file bytes, symlinks excluded).

## Outputs
Imported as `from vqlab.util.<module> import <name>`.

## Rules that bite
- Stdlib only (plus `vqlab.config` if ever needed). Nothing here may import a stage: `tests/test_layering.py` enforces it.
- One home per helper. Before writing a private `_sha` / `_log` / `_now` / `_slug` / `_dir_bytes`, import the one here. The old private names in provenance, numerics, doctor, pin, reselect, release_prep, sz_*, geo_build, runlog, run_queue, reserve, queue_eta, mcp_server and scratch_cmd now point here and are kept for one release.

## Files
- `clock.py`: time stamps and filename-safe slugs
- `fs.py`: filesystem measurements (`dir_bytes`)
- `hashing.py`: content hashes: files, bytes, git blobs, `fingerprint`
- `log.py`: progress lines for long-running tools, flushed
