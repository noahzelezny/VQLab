# core/box/ — this machine and its services

The GPU lease every GPU tenant takes in turn, the exo cluster's API as the
GPU gate reads it, the Knurlogic HTTP/job client, and the Knurlogic
architecture finder behind `vqlab.use_knurlogic_arch()`.

## Rules that bite
- Part of the core layer (L1): imports nothing from `vqlab.family` or above.
- Machine-specific values (lease path, hosts) come from `vqlab.config`, never from code.
- Old names `vqlab.core.<module>` (and `vqlab.family.arch` for `arch_finder`)
  resolve here for one release (`_layout.DEPRECATED`).

## Files
- `gpu_lease.py`: the GPU lease file every GPU tenant flocks in turn (`lease_path`, `lease_holder`; box_quiet, ops/, queue/ and the MCP server)
- `exo.py`: the exo cluster's HTTP API as the GPU gate reads it: is an instance placed?
- `knurlogic_client.py`: Knurlogic HTTP/job client helpers (`post`, `entry`, `ready`, ...; bench and gate arms)
- `arch_finder.py`: the Knurlogic architecture meta-path finder behind `vqlab.use_knurlogic_arch()` (`install_arch_finder()`)
