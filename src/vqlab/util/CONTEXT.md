# util/ — Small helpers every stage shares

## Inputs
Paths, bytes, strings. Nothing model-shaped.

## Process
`hashing.py` (`sha256_file` with head/algo/missing_ok, `sha256_bytes`, `git_blob_sha1`, `fingerprint(dir)`: an input directory's cheap identity), `clock.py` (`now()` aware UTC datetime, `now_iso()`, `slug()`), `log.py` (`log()`, `make_log(prefix, stamp)`: flushed progress lines), `fs.py` (`dir_bytes()`: regular-file bytes, symlinks excluded; `GIB`; `atomic_path` / `write_text_atomic` / `write_json_atomic` / `copy_atomic`: whole-file writes with a unique temp beside the target; `read_jsonl`), `errors.py` (`UserError`: a refused request, the GUI's 400), `geo.py` (`geo_key(d, k)`: dD-KK).

## Outputs
Imported as `from vqlab.util.<module> import <name>`.

## Rules that bite
- Stdlib only (plus `vqlab.config` if ever needed). Nothing here may import a stage: `tests/test_layering.py` enforces it.
- One home per helper. Before writing a private `_sha` / `_log` / `_now` / `_slug` / `_dir_bytes`, import the one here. The deprecated compat modules (run_queue, mcp_server, scratch_cmd) still answer to their old private names.

## Files
- `clock.py`: time stamps and filename-safe slugs
- `errors.py`: `UserError`, a request refused as asked (the GUI answers 400)
- `fs.py`: filesystem measurements and atomic writes, JSON lines
- `geo.py`: `geo_key`, a geometry's dD-KK name
- `hashing.py`: content hashes: files, bytes, git blobs, `fingerprint`
- `log.py`: progress lines for long-running tools, flushed
