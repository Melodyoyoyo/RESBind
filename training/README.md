# Model training

This directory contains the RESBind [training implementation](train_v6.py).

Install the training dependencies from the repository root:

```sh
python -m pip install ".[training]"
```

Prepare split-specific feature tables, per-protein embeddings, and structural contact files. The training entry point accepts the following paths:

| Argument | Input |
| --- | --- |
| `--feature-root` | Directory containing the split-specific feature tables |
| `--embedding-dir` | Directory containing per-protein embeddings |
| `--contact-dir` | Directory containing ESMFold contact files |
| `--out-dir` | Directory for training outputs |

Use the entry point's `--help` option for available training arguments. See the [training and evaluation guide](../skills/resbind/references/labeled-data.md) for dataset preparation.
