# Prediction and feature preparation

## Install and run the example

Clone the [RESBind repository](https://github.com/Melodyoyoyo/RESBind) or extract its ZIP archive. Create a dedicated Python environment and install from the repository root:

```sh
python -m pip install .
resbind predict --fasta examples/P07014/sequence.fasta --pdb examples/P07014/structure.pdb --netsurfp-csv examples/P07014/netsurfp.csv --prott5-npz examples/P07014/prott5.npz --device cpu --out-dir results/P07014
```

Keep the `models/` directory alongside the code. When running outside the repository root, set `RESBIND_BUNDLE_DIR` to the absolute path of the bundled multi-model directory. `RESBIND_JOB_ROOT` and `RESBIND_CACHE_ROOT` select writable job and cache locations.

## Prepare inputs

Use one protein FASTA record per job, with 20–1,024 residues. All features must match that sequence and its residue order. The sequence parser maps B, Z, U, O, and J to X.

The workflow uses ProtT5 residue embeddings, an ESMFold structure, and NetSurfP surface features. Existing sequence-matched features can be reused. For new features, choose a CUDA machine using the [compute guide](compute.md).

### ProtT5 and ESMFold

Install feature-generation dependencies on the CUDA machine:

```sh
python -m pip install ".[features]"
```

Obtain local model snapshots for `Rostlab/prot_t5_xl_uniref50` and `facebook/esmfold_v1`. Run the stages sequentially, using the snapshot directories and new output directories:

```sh
python -m resbind_server.feature_stage --stage prott5 --fasta input.fasta --model-dir prott5-model --out-dir features-prott5
python -m resbind_server.feature_stage --stage esmfold --fasta input.fasta --model-dir esmfold-model --out-dir features-esmfold
```

These commands produce `features-prott5/prott5.npz`, `features-esmfold/structure.pdb`, and provenance records. GPU memory requirements depend on sequence length. Check the outputs for sequence alignment and finite values before prediction.

### NetSurfP

Submit the protein FASTA to the [NetSurfP-3.0 service](https://services.healthtech.dtu.dk/services/NetSurfP-3.0/) and export the results as CSV. For assistant-operated submission, obtain the user's authorization to send the sequence to the service. Save the CSV and its job reference with the input files.

## Run prediction

From the repository root:

```sh
resbind predict --fasta input.fasta --pdb features-esmfold/structure.pdb --netsurfp-csv netsurfp.csv --prott5-npz features-prott5/prott5.npz --device cpu --out-dir results/query
```

Compatible AI assistants can use the [local MCP interface](mcp.md) with the same prepared inputs.

## Read the results

- `residue_predictions.csv`: probabilities and classifications for metal ions, nucleic acids, and small molecules.
- `pockets.csv`: ranked spatial candidates with residue positions, scores, and centers.
- `pockets_independent_seeds.csv`: per-model candidates from the default bundle.
- `structure.pdb`: the structure used for prediction.
- `summary.json`: task summaries and input metadata.
- `resbind_results.zip`: the result bundle.

Residue numbering starts at 1 and follows the FASTA sequence. `candidate_type` distinguishes multi-residue pockets from isolated residues. Use a separate output directory for each protein.
