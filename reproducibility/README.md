# Benchmark evaluation

This directory contains evaluation scripts and protocol definitions for RESBind's residue-level and binding-site localization benchmarks.

## Residue-level evaluation

`evaluate_table1.py` evaluates saved residue predictions. The task thresholds and aggregation rules are recorded in [paper_protocols.json](paper_protocols.json). Results are computed for each model and summarized as mean and sample standard deviation.

## Binding-site localization

Run the evaluator with per-model prediction tables, reference sites, and matching ESMFold structures:

```sh
python reproducibility/evaluate_pockets.py --seed41 seed41_test_predictions.csv --seed42 seed42_test_predictions.csv --seed43 seed43_test_predictions.csv --sites reference_sites.csv --pdb-dir esmfold_pdb --out-dir pocket_replay
```

### Input format

| Input | Required fields |
| --- | --- |
| Prediction CSVs | `peptide_id`, `center_pos`, `prob_metal`, `prob_nuclear`, `prob_small`, `mask_metal`, `mask_nuclear`, `mask_small` |
| Reference-site CSV | `task`, `peptide_id`, `pocket_id`, `pocket_residues` |
| Structure directory | Single-chain PDB files with contiguous sequence-position numbering |

The `task` field uses `metal`, `nuclear`, or `small`. Reference residue positions are 1-based. Use the same evaluation cohort, residue keys, and masks for all prediction tables.

### Outputs

The evaluator writes per-site distances, per-model metrics, aggregate means and sample standard deviations, and input checksums. Localization uses same-protein Top-N matching with a strict center-distance criterion of less than 10 Å. Candidate sets are evaluated independently for each model.

See [paper_protocols.json](paper_protocols.json) for the complete evaluation configuration and [all_independent_pocket_replay.json](all_independent_pocket_replay.json) for the recorded benchmark replay.
