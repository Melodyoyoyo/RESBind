# Local MCP

## Invocation policy

Prefer MCP for input inspection and saved-feature prediction when the agent client
supports MCP and the user authorizes its configuration. Respect an explicit CLI
preference. Installation of the MCP package alone does not configure a client or
prove that a prediction used MCP.

If MCP is unavailable or cannot be configured, explain the reason and continue
with the documented CLI when command execution is available and authorized.
If neither route is available, provide a manual handoff. Do not repeatedly change
global client settings or treat a successful CLI run as a failed prediction.
After a failed or interrupted MCP call, inspect its output and process state before
retrying via CLI to avoid duplicate jobs. Preserve logs and use a new output
directory for any necessary retry. Reuse validated features; do not regenerate
ProtT5, ESMFold or NetSurfP outputs merely to change the invocation method.

Write the actual execution route in an English `run_record.json` alongside the
outputs. Preserve existing provenance and use these additional fields:

- `prediction_interface`: `mcp` or `cli`, based on the actual prediction call.
- `mcp_transport`: `stdio` for MCP, otherwise null.
- `mcp_tools_called`: tool names actually invoked, or an empty list.
- `cli_fallback_reason`: the reason for fallback, or null when not applicable.
- `execution_evidence`: paths to retained tool-call records or command logs.

If the call route cannot be established retrospectively, use `unknown` for
`prediction_interface`; do not infer MCP use from a results ZIP or package install.
Do not store credentials in execution records. An MCP prediction delegates to the
CLI internally; record `mcp` when the agent actually invoked `predict_saved`.
Upstream feature generation remains a separate browser/terminal workflow, not an
MCP capability. Distinguish protocol-level tests from an actual agent-client run.

## Configuration and tools

Install the checkout with `python -m pip install ".[mcp]"` in Python >=3.10.
SDK 1.x is explicitly pinned; this is a local stdio server, not a public HTTP API.

Configure the chosen agent client, with the user's permission, to launch the
absolute environment Python executable with arguments:

```text
-I -m resbind_server.mcp_server
```

Pass these environment variables, resolved to actual local absolute paths:

- `RESBIND_MCP_WORKSPACE`: existing dedicated input/output directory.
- `RESBIND_BUNDLE_DIR`: trusted checkout's `models/v6_bidirectional_ensemble`.

Use that client's documented MCP configuration format; do not overwrite unrelated
servers/settings. The server runs with the user's OS permissions. Its path checks
are not an OS sandbox; keep the workspace owner-controlled, not writable by
untrusted processes. Do not expose this process as a network service.

Tools:

- `check_environment()`: Python, PyTorch and visible CUDA devices; no model load.
- `inspect_fasta(fasta, mode, pdb?, netsurfp_csv?, prott5_npz?)`: validate files
  inside the configured workspace. Modes predict/evaluate/train only route intent.
- `predict_saved(fasta, pdb, netsurfp_csv, prott5_npz)`: synchronous CPU prediction,
  one job at a time, all supplied inputs mandatory, 180-second timeout, unique
  output directory. No implicit downloads or heavy upstream stages.

After initialization, list tools, inspect the example, run prediction, and compare
all three tasks to `expected_predictions.csv`. The supplied reproducible protocol
test is `python tests/mcp_acceptance.py .` from the checkout, using the installed
package. It creates a fresh local run; it does not configure the agent client.

The complete Skill combines these tools with the agent's browser and terminal.
Do not call missing job-queue, training, progress-query or feature-generation MCP
tools: those are not implemented. Training is not performed by `predict_saved`.
