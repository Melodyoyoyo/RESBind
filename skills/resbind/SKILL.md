---
name: resbind
description: Predict protein binding residues and sites for metal ions, nucleic acids, and small molecules using RESBind. Prepare sequence features, run CLI or MCP prediction, and return result files.
---

# RESBind

Repository: https://github.com/Melodyoyoyo/RESBind

## Choose a workflow

- **Prediction:** follow [prediction.md](references/prediction.md) to validate a sequence, prepare features, and run inference.
- **MCP integration:** follow [mcp.md](references/mcp.md) to configure the local server and use its prediction tools.
- **Training or evaluation:** follow [labeled-data.md](references/labeled-data.md) for dataset requirements and entry points.

For a new installation, clone the repository or download its latest main-branch ZIP archive. For an existing installation, check for updates before prediction unless the user requests a specific revision. Update a clean main-branch checkout with `git pull --ff-only`; preserve local changes and use a separate checkout when needed. Read the Skill and referenced instructions from the updated repository, including when setup began with the website Skill ZIP. Install RESBind in a dedicated Python environment and run the bundled P07014 example. After an update, reinstall the package in that environment and restart any running MCP server so it loads the current model configuration.

## Inputs and compute

Each prediction accepts one protein sequence of 20–1,024 residues. Check that the ESMFold structure, ProtT5 embeddings, and NetSurfP CSV match the input sequence and residue order. Report any sequence normalization to the user.

Use CPU inference for prepared features. For missing ProtT5 or ESMFold features, select a CUDA resource using [compute.md](references/compute.md). Scheduled clusters are covered in [hpc.md](references/hpc.md). Obtain NetSurfP results from the official service with the user's authorization to submit the sequence.

Use the bundled model configuration for prediction. Keep checkpoints and scalers together. Preserve user-provided data splits and labels when handling evaluation or training requests.

## Results

Return residue predictions for all three ligand classes, spatial binding-site candidates, the structure, summary, and result ZIP. Explain that positions use 1-based sequence numbering and that isolated residues are listed separately from multi-residue pockets.

Check sequence length, residue order, finite probabilities, and the presence of the output files. Keep code revision, input checksums, feature sources, and execution logs with the local run. For multiple proteins, use separate output directories.
