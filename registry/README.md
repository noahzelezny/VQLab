# registry/ -- moved

The artifact registry no longer lives in the repo. `vqlab lab registry`
reads and writes `<records>/registry/*.jsonl`, where `<records>` is the
`records` location (`$VQLAB_RECORDS`, `[paths] records` in the vqlab config,
else `~/.local/share/vqlab`); `vqlab lab config` prints it.

If an older checkout still holds `registry/artifacts.jsonl` or
`registry/hub.jsonl`, copy them into `<records>/registry/` (see CHANGELOG,
"Migration: records moved"). `vqlab lab doctor` warns while they sit here.

Usage:

    vqlab lab registry list --grep <name>
    vqlab lab registry hub --all      # local vs Hub, metadata only
