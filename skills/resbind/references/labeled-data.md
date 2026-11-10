# Training and evaluation

RESBind includes training code and scripts for evaluating residue predictions and spatial binding-site candidates.

## Dataset preparation

Prepare protein sequences, per-residue labels for the relevant ligand classes, missing-label masks, and sequence-to-structure mappings. Keep the protein identifiers and residue positions consistent across feature tables, embeddings, contacts, and labels.

Use explicit training, validation, and test partitions. Fit preprocessing on the training partition and use the validation partition for model selection. Preserve missing-label masks when computing metrics.

## Training

Install the training dependencies:

```sh
python -m pip install ".[training]"
```

The training entry point is in the repository's `training/` directory. Its command-line help describes the split-specific feature tables, embedding and contact directories, output location, and training options. Supply the prepared data paths when starting a run.

## Evaluation

Use the [benchmark evaluation scripts](../../../reproducibility/README.md) to evaluate residue-level predictions or replay binding-site localization from saved probabilities and structures. Keep the evaluation cohort, annotations, and coordinate mappings consistent across compared methods.

Report task-specific metrics with the dataset definition and aggregation procedure. Store the input files and run configuration alongside the results.
