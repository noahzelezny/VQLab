"""vqlab mcp — the lab's MCP server, as a package.

A thin adapter over the library: every tool maps its JSON arguments onto
vqlab.api (or vqlab.ops, where the api deliberately has no such call) and
returns the result. No lab logic lives here.

    server.py          stdio JSON-RPC loop + the tool table (name -> fn, schema)
    common.py          ToolError (= vqlab.ops.errors.LabError), old helper names
    tools_storage.py   adapters for the read-only storage tools: where_is, list_artifacts,
                       artifact_config, read_doc, disk_free
    tools_runs.py      run / status / stop / list_runs / gpu_state /
                       queue_status, and the detached supervisor
    tools_findings.py  next_f_number (reserve/release), findings_tail,
                       findings_append

`vqlab.agents.mcp_server` is a deprecated re-export of these names.
"""
