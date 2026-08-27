#!/usr/bin/env python3
"""Delete the artifacts listed in deletion_plan.json. Dry-run by default.

    ./delete_artifacts.py            # show what would go, delete nothing
    ./delete_artifacts.py --execute  # actually delete

Refuses to run unless every KEEP directory is present and its .safetensors
total matches its exo card, so a stale plan cannot delete something the
picker still points at. (An earlier plan, generated before VQ-2.6bpw was
renamed into place, had it on the delete list -- hence this check.)
"""
import json, os, shutil, sys

ROOT = "/Volumes/Thunderbay SSD/Exo Models"
CARDS = os.path.expanduser("~/exo/resources/inference_model_cards")


def st_total(d):
    return sum(os.path.getsize(os.path.join(d, f))
               for f in os.listdir(d) if f.endswith(".safetensors"))


def main():
    go = "--execute" in sys.argv
    plan = json.load(open("deletion_plan.json"))
    keep, dele = plan["keep"], plan["delete"]

    print(f"KEEP {len(keep)} / DELETE {len(dele)}\n")
    for k in keep:
        d = os.path.join(ROOT, k)
        if not os.path.isdir(d):
            sys.exit(f"ABORT: keep-set directory missing: {k}")
        card = os.path.join(CARDS, k + ".toml")
        if os.path.exists(card):
            want = int([l for l in open(card) if l.startswith("in_bytes")][0].split("=")[1])
            got = st_total(d)
            if want != got:
                sys.exit(f"ABORT: {k} card says {want}, disk has {got}")
            print(f"  keep OK  {k}  (card matches disk)")
        else:
            print(f"  keep OK  {k}  (no exo card)")

    total = 0
    for n in dele:
        d = os.path.join(ROOT, *n.split("/"))
        # A few entries are symlinks into the hub caches (which are themselves
        # on the delete list). rmtree REFUSES a symlink -- that aborts the whole
        # run on the first one. Unlink it and count zero bytes: the space is
        # freed when the cache it points into is removed, and counting it here
        # would double-count.
        if os.path.islink(d):
            tgt = os.readlink(d)
            if go:
                os.unlink(d)
                print(f"  unlinked (symlink -> {tgt})  {n}", flush=True)
            else:
                print(f"  would unlink (symlink -> {tgt})  {n}")
            continue
        if not os.path.isdir(d):
            continue
        sz = sum(os.path.getsize(os.path.join(r, f))
                 for r, _, fs in os.walk(d) for f in fs
                 if os.path.exists(os.path.join(r, f)))
        total += sz
        if go:
            shutil.rmtree(d)
            print(f"  deleted {sz/2**30:8.1f} GiB  {n}", flush=True)
        else:
            print(f"  would delete {sz/2**30:8.1f} GiB  {n}")
    print(f"\n{'FREED' if go else 'WOULD FREE'} {total/2**40:.2f} TiB")
    if not go:
        print("dry run — nothing deleted. re-run with --execute")


if __name__ == "__main__":
    main()
