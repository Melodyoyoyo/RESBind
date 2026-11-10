# RESBind

RESBind predicts protein residues that bind metal ions, nucleic acids, and small molecules. It combines protein language model representations, surface features, and structural contacts to produce residue-level predictions and spatial binding-site candidates for all three ligand classes.

This repository provides pretrained models, a command-line prediction workflow, example inputs, training code, and evaluation scripts. An optional Model Context Protocol (MCP) interface supports prediction through compatible AI assistants.

## Installation

Use Python 3.9 or later. The optional MCP interface requires Python 3.10 or later.

```sh
git clone https://github.com/Melodyoyoyo/RESBind.git
cd RESBind
python -m venv .venv
```

Activate the environment on Linux or macOS:

```sh
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install RESBind:

```sh
python -m pip install .
```

Run the commands below from the repository root, with the bundled `models/` directory in place.

## Quick start

The P07014 example includes a protein sequence and precomputed ESMFold, ProtT5, and NetSurfP features. Run the example on CPU:

```sh
resbind predict --fasta examples/P07014/sequence.fasta --pdb examples/P07014/structure.pdb --netsurfp-csv examples/P07014/netsurfp.csv --prott5-npz examples/P07014/prott5.npz --device cpu --out-dir results/P07014
```

## Predict a protein

Prepare the following files for the same protein sequence:

| Input | Format | Content |
| --- | --- | --- |
| Protein sequence | FASTA | One sequence of 20–1,024 residues |
| Predicted structure | PDB | ESMFold structure matching the sequence |
| Surface features | CSV | NetSurfP predictions |
| Residue embeddings | NPZ | ProtT5 embeddings |

```sh
resbind predict --fasta input.fasta --pdb structure.pdb --netsurfp-csv netsurfp.csv --prott5-npz prott5.npz --device cpu --out-dir results/query
```

For a new sequence, follow the [feature preparation guide](skills/resbind/references/prediction.md). ProtT5 and ESMFold feature generation uses a CUDA GPU. NetSurfP results can be downloaded as CSV from the official service.

## Outputs

Each prediction produces the following files:

| File | Content |
| --- | --- |
| `residue_predictions.csv` | Binding probabilities and residue classifications for all three ligand classes |
| `pockets.csv` | Ranked spatial candidates, residue positions, scores, and centers |
| `pockets_independent_seeds.csv` | Candidate sites from each model in the default bundle |
| `structure.pdb` | Structure used for prediction |
| `summary.json` | Prediction summary and input metadata |
| `resbind_results.zip` | Downloadable result bundle |

Residue positions use 1-based sequence numbering. Spatial candidates are labeled as multi-residue pockets or isolated residues.

## Documentation

- [Prediction and feature preparation](skills/resbind/references/prediction.md)
- [Compute options](skills/resbind/references/compute.md)
- [MCP integration](skills/resbind/references/mcp.md)
- [AI assistant workflow](skills/resbind/SKILL.md)
- [Training and evaluation](skills/resbind/references/labeled-data.md)
- [Benchmark evaluation scripts](reproducibility/README.md)

## License

See the [MIT License](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md).
