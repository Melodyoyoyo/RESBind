# MCP integration

RESBind provides a local stdio MCP server for input validation and CPU prediction with prepared features.

## Installation

Use Python 3.10 or later and install from the repository root:

```sh
python -m pip install ".[mcp]"
```

Use a dedicated environment for MCP. The optional Gradio `web` interface uses a separate dependency environment.

## Client configuration

Configure your MCP client to launch the environment's absolute Python executable with these arguments:

```text
-I -m resbind_server.mcp_server
```

Set the following environment variables to absolute paths:

| Variable | Value |
| --- | --- |
| `RESBIND_MCP_WORKSPACE` | An existing directory containing the prediction inputs and outputs |
| `RESBIND_BUNDLE_DIR` | The bundled multi-model directory under `models/` |

The server uses local files in the configured workspace. Keep the server on the local stdio connection provided by the client.

## Tools

| Tool | Function |
| --- | --- |
| `check_environment()` | Report the Python, PyTorch, and CUDA environment |
| `inspect_fasta(fasta, mode, pdb?, netsurfp_csv?, prott5_npz?)` | Validate a sequence and supplied features |
| `predict_saved(fasta, pdb, netsurfp_csv, prott5_npz)` | Run CPU prediction and return output paths |

Place the matching FASTA, ESMFold PDB, NetSurfP CSV, and ProtT5 NPZ inside the workspace. Call `inspect_fasta` and then `predict_saved`. Predictions run one job at a time in separate output directories.

Prepare missing features using the [feature preparation guide](prediction.md). The [CLI workflow](prediction.md#run-prediction) provides an alternative interface for the same inputs and outputs.

## Integration check

From the installed checkout:

```sh
python tests/mcp_acceptance.py .
```
