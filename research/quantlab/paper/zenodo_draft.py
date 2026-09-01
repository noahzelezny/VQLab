#!/usr/bin/env python3
"""Create the Zenodo DRAFT for the paper and reserve its DOI. Publishes NOTHING.

    ZENODO_TOKEN=<your token> python3 zenodo_draft.py

The token stays in your shell env — this script never writes it anywhere.
Get one at zenodo.org -> account -> Applications -> Personal access token,
scope "deposit:write". The draft it creates is private until YOU press
Publish on the Zenodo page; the reserved DOI is real and final the moment
it is reserved, so it can be stamped into the paper before anything goes
public.

Prints: the draft URL (for your review + the eventual Publish click) and the
reserved DOI (for stamping). Re-running does not duplicate: it looks for an
existing draft with this title first.
"""
import json, os, sys, urllib.request

API = "https://zenodo.org/api"
TITLE = ("Data-Free Vector Quantization Beats Affine Quantization "
         "at Matched Bytes Below 6 Bits")

def req(method, url, token, data=None, ctype="application/json"):
    r = urllib.request.Request(url + ("&" if "?" in url else "?") +
                               "access_token=" + token, method=method)
    if data is not None and ctype:
        r.add_header("Content-Type", ctype)
    body = json.dumps(data).encode() if isinstance(data, dict) else data
    with urllib.request.urlopen(r, body) as resp:
        return json.load(resp) if resp.length != 0 else {}

def main():
    tok = os.environ.get("ZENODO_TOKEN")
    if not tok:
        # fall back to the macOS Keychain (add once with:
        #   security add-generic-password -a "$USER" -s zenodo-token -w
        # the -w with no value PROMPTS for the secret, so it never lands in
        # shell history). Encrypted at rest; no plaintext in dotfiles.
        import subprocess
        r = subprocess.run(["security", "find-generic-password",
                            "-s", "zenodo-token", "-w"],
                           capture_output=True, text=True)
        tok = r.stdout.strip()
    if not tok:
        sys.exit("no token: set ZENODO_TOKEN, or store one in the Keychain --\n"
                 '  security add-generic-password -a "$USER" -s zenodo-token -w')

    # reuse an existing draft rather than minting a second DOI
    drafts = req("GET", f"{API}/deposit/depositions?status=draft&size=50", tok)
    dep = next((d for d in drafts if d["title"] == TITLE), None)
    if dep is None:
        dep = req("POST", f"{API}/deposit/depositions", tok, {})
        print("created new draft")
    else:
        print("reusing existing draft", dep["id"])
        # listing entries are slim records without links.bucket -- re-fetch full
        dep = req("GET", f"{API}/deposit/depositions/{dep['id']}", tok)

    doi = dep["metadata"]["prereserve_doi"]["doi"]
    meta = {"metadata": {
        "title": TITLE,
        "upload_type": "publication",
        "publication_type": "preprint",
        "description": (
            "<p>Vector quantization of three models — Qwen3.5-397B-A17B, "
            "Qwen3.6-35B-A3B (mixture-of-experts), and the dense Qwen3.8-27B — "
            "measured against uniform and hand-tuned mixed-bit-depth affine "
            "builds at matched or smaller file sizes. Codebooks are fit by "
            "k-means over the weight tensors alone: no calibration corpus, no "
            "activations, no teacher model. Below roughly 5 bits per weight the "
            "data-free VQ builds outperform the affine builds; the crossover "
            "where affine overtakes is bracketed at 4.5–6.0 bits per weight. "
            "Thirteen artifacts are published with pinned revisions.</p>"
            "<p>Models: https://huggingface.co/TheDrainFlorist</p>"),
        "creators": [{"name": "Zelezny, Noah"}],
        "license": "cc-by-4.0",
        "prereserve_doi": True,
        "keywords": ["quantization", "vector quantization", "large language models",
                     "Apple Silicon", "MLX", "model compression"],
        "related_identifiers": [
            {"identifier": "https://huggingface.co/TheDrainFlorist",
             "relation": "isSupplementedBy", "scheme": "url"},
        ],
    }}
    req("PUT", f"{API}/deposit/depositions/{dep['id']}", tok, meta)

    bucket = dep["links"]["bucket"]
    for fn in ("below-six-bits.pdf", "paper.html"):
        with open(fn, "rb") as fh:
            req("PUT", f"{bucket}/{fn}", tok, fh.read(),
                ctype="application/octet-stream")
        print("uploaded", fn)

    print("\ndraft URL :", dep["links"]["html"])
    print("DOI (reserved, stamp this):", doi)
    print("\nNothing is published. Review the draft, and press Publish only on")
    print("publish day, after the DOI is stamped into the paper and the site.")

if __name__ == "__main__":
    main()
