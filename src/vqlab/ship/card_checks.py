"""The refusals behind `vqlab card`: a card never states numbers that were
not measured on this artifact.

check_kl   refuses a KL JSON whose fingerprint for --this is not the
           artifact's (unless the only difference is a sidecar holding no
           text tensors: text_match names it), mixed builds, or a scorer
           that saw artifacts change during the run.
"""
from __future__ import annotations

import json
import pathlib
import sys

from vqlab.records import provenance
from vqlab.ship import card_tables
from vqlab.util.fs import read_json
from vqlab.core.artifacts.artifact import Artifact


class Refuse(SystemExit):
    def __init__(self, msg):
        print(f"vqlab card: REFUSED: {msg}", file=sys.stderr)
        super().__init__(2)


def check_kl(kl_paths, this: str, artifact: pathlib.Path) -> dict:
    """Refusals, and the KL JSON facts the card states. Returns
    {"fingerprint": stamped-or-None, "tokens": {corpus: n}, "variants": set}."""
    have = provenance.shard_fingerprint(artifact)
    stamped, found, keys, tokens = set(), False, set(), {}
    for p in kl_paths:
        d = read_json(p)
        if d.get("changed_during_run"):
            raise Refuse(f"{p}: the scorer recorded artifacts changing during the run: "
                         f"{d['changed_during_run']}")
        for rung, per in d["table"].items():
            for c, row in per.items():
                keys.add((card_tables.measured_with([row.get("numerics")]),
                          row.get("scorer_variant")))
                if rung == this:
                    fp = (row.get("measured") or {}).get("fingerprint")
                    if fp:
                        stamped.add(fp)
                    tokens[c] = row.get("kl_positions") or row.get("tokens")
        if this in d["table"]:
            found = True
            fp = ((d.get("measured") or {}).get(this) or {}).get("fingerprint")
            if fp:
                stamped.add(fp)
    if not found:
        raise Refuse(f"rung {this!r} is in none of the KL files")
    if len(keys) > 1:
        raise Refuse("KL measured under MIXED builds (not one harness): "
                     + " | ".join(f"{b} variant={v}" for b, v in sorted(keys, key=str)))
    bad = sorted(s for s in stamped if s != have)
    added = None
    if bad and len(stamped) == 1:
        added = text_match(artifact, bad[0])
    if bad and added is None:
        raise Refuse(f"KL for {this!r} was measured on fingerprint {', '.join(bad)}, "
                     f"but {artifact} is {have}: these numbers are not this artifact's")
    return {"fingerprint": next(iter(stamped)) if stamped else None, "tokens": tokens,
            "variant": next(iter(keys))[1], "added_since": added or []}


def _fp(files) -> str:
    import hashlib
    import os
    return hashlib.sha256(json.dumps(
        {f.name: [os.path.getsize(os.path.realpath(f)), provenance._sha(os.path.realpath(f),
                                                                       provenance.HEAD)]
         for f in files}, sort_keys=True).encode()).hexdigest()[:16]


def text_match(artifact: pathlib.Path, want: str):
    """The scored fingerprint can predate a sidecar that holds NO text
    weights (a grafted vision tower, an MTP head). If dropping some set of
    such files reproduces `want`, the text bytes are the scored ones:
    return those file names. Else None. A file holding any text tensor is
    never dropped, so a changed weight still refuses."""
    from itertools import combinations
    from vqlab.core.artifacts.artifact import read_header, tensor_class
    files = sorted(artifact.glob("*.safetensors"))
    try:
        listed = set(Artifact(artifact).index.values())
    except (OSError, KeyError, ValueError):
        listed = set()
    # an MTP sidecar (mtp-head*.safetensors outside the index, as `size`
    # counts it) stores its keys without the mtp. prefix (block.*, fc.*), so
    # it is recognised by that rule, not by tensor_class
    side = [f for f in files
            if (f.name.startswith("mtp-head") and f.name not in listed)
            or all(tensor_class(k) != "text" for k in read_header(f))]
    for r in range(1, len(side) + 1):
        for drop in combinations(side, r):
            if _fp([f for f in files if f not in drop]) == want:
                return [f.name for f in drop]
    return None
