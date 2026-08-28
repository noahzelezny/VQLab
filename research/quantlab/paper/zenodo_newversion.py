#!/usr/bin/env python3
"""Open a NEW VERSION draft of the published record and reserve its DOI.
Publishes NOTHING.

    python3 zenodo_newversion.py --parent <record_id> [--stage]

Zenodo versioning: POST .../actions/newversion forks the published record
into a fresh draft that INHERITS the previous version's files. The draft
carries its own prereserve_doi, which is real and final the moment it is
reserved -- so it can be stamped into the paper before anything is public.

--stage additionally clears the inherited files and uploads the current
below-six-bits.pdf + paper.html from this directory. Do that only AFTER the
new DOI is stamped and the artifacts rebuilt, or the draft ships a PDF whose
front matter cites the PREVIOUS version.

Token: $ZENODO_TOKEN, else the macOS Keychain item "zenodo-token".
"""
import argparse, json, os, subprocess, sys, urllib.request

API = "https://zenodo.org/api"

def token():
    t = os.environ.get("ZENODO_TOKEN")
    if not t:
        r = subprocess.run(["security", "find-generic-password",
                            "-s", "zenodo-token", "-w"],
                           capture_output=True, text=True)
        t = r.stdout.strip()
    if not t:
        sys.exit('no token: set ZENODO_TOKEN or store Keychain item "zenodo-token"')
    return t

def req(method, url, tok, data=None, ctype="application/json"):
    r = urllib.request.Request(url + ("&" if "?" in url else "?") +
                               "access_token=" + tok, method=method)
    if data is not None and ctype:
        r.add_header("Content-Type", ctype)
    body = json.dumps(data).encode() if isinstance(data, dict) else data
    with urllib.request.urlopen(r, body) as resp:
        return json.load(resp) if resp.length != 0 else {}

ap = argparse.ArgumentParser()
ap.add_argument("--parent", help="published record id to fork a new version from")
ap.add_argument("--draft", help="stage into THIS existing draft id instead of "
                                "forking. Use when the version draft already "
                                "exists -- Zenodo refuses a second fork, and "
                                "the parent's latest_draft link points at the "
                                "PARENT, so resolving it walks you onto the "
                                "published record and tries to delete its "
                                "files (403, 2026-08-27).")
ap.add_argument("--stage", action="store_true",
                help="replace inherited files with the current local build")
a = ap.parse_args()
if not (a.parent or a.draft):
    sys.exit("need --parent (to fork) or --draft (to stage into an open draft)")
tok = token()

if a.draft:
    draft_id = a.draft
else:
    nv = req("POST", f"{API}/deposit/depositions/{a.parent}/actions/newversion", tok)
    draft_id = nv["links"]["latest_draft"].rstrip("/").split("/")[-1]

dep = req("GET", f"{API}/deposit/depositions/{draft_id}", tok)
# HARD GUARD: never mutate a published record. Deleting the files off a live
# DOI is not recoverable by re-running anything.
if dep.get("submitted") or dep.get("state") == "done":
    sys.exit(f"REFUSING: deposition {draft_id} is PUBLISHED (state="
             f"{dep.get('state')}). Staging would delete files from a live "
             f"DOI. Pass the unsubmitted draft id with --draft.")
doi = dep["metadata"]["prereserve_doi"]["doi"]
print(f"draft id  : {draft_id}")
print(f"draft URL : {dep['links']['html']}")
print(f"NEW DOI   : {doi}")

if a.stage:
    for f in dep.get("files", []):
        req("DELETE", f"{API}/deposit/depositions/{draft_id}/files/{f['id']}", tok)
        print("removed inherited", f["filename"])
    bucket = dep["links"]["bucket"]
    for fn in ("below-six-bits.pdf", "paper.html"):
        with open(fn, "rb") as fh:
            req("PUT", f"{bucket}/{fn}", tok, fh.read(),
                ctype="application/octet-stream")
        print("uploaded", fn)
    print("\nStaged. NOTHING IS PUBLISHED -- press Publish on the draft page.")
else:
    print("\nDOI reserved, no files touched. Stamp it, rebuild, then re-run "
          "with --stage.")
