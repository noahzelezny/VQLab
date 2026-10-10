"""Small dependency-free helpers every stage shares (stdlib only).

    hashing   sha256_file, sha256_bytes, git_blob_sha1, fingerprint(dir)
    clock     now(), now_iso(), slug()
    log       log(), make_log(prefix)
    fs        dir_bytes()

One home per helper: before writing a private `_sha` / `_log` / `_now`,
import it from here.
"""
