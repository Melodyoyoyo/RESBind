# Local MCP

Install the checkout with `python -m pip install ".[mcp]"` in Python >=3.10.
SDK 1.x is explicitly pinned; this is a local stdio server, not a public HTTP API.

Configure the chosen agent client, with the user's permission, to launch the
absolute environment Python executable with arguments:

```text
-I -m resbind_server.mcp_server
```

Pass these environment variables, resolved to actual local absolute paths:

- `RESBIND_MCP_WORKSPACE`: existing dedicated input/output directory.
- `RESBIND_BUNDLE_DIR`: trusted checkout's `models/v6_bidirectional_seed42`.

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
