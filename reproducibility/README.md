# Paper protocols and three-seed models

The three Full models are in `models/v6_bidirectional_seed41`, `seed42` and
`seed43` (each directory uses the full `v6_bidirectional_seed` prefix).
Each bundle includes its own checkpoint, manifest and four verified scalers.
The existing seed42 model bundle and its stored thresholds are unchanged.
The application default is now `models/v6_bidirectional_ensemble`.
`prob_*`, `threshold_*` and `pred_*` retain the seed42 residue operating point.
`prob_*_seed41/42/43` export each seed's probabilities for independent residue
evaluation. `pocket_prob_*` contains 0.4/0.2/0.4 weighted ensemble probabilities
and is used for pocket selection, centers and ranking.
Select a different bundle with the absolute `RESBIND_BUNDLE_DIR` environment variable.
No refiner is used.

## Table 1: independent seed evaluation

Run inference separately with each bundle on exactly the original test residues,
using the same frozen input features. Preserve `peptide_id`, `center_pos`, task
labels and task masks in each prediction CSV. The expected columns are
`label_metal`, `label_nuclear`, `label_small`, corresponding `mask_*` and `prob_*`.
The CLI's public task names `nucleic_acid` and `small_molecule` need mapping to
`nuclear` and `small` when preparing benchmark CSVs.

Install evaluation dependencies with `python -m pip install ".[training]"`.
From the repository root run:

```sh
python reproducibility/evaluate_table1.py --seed41 seed41_test_predictions.csv --seed42 seed42_test_predictions.csv --seed43 seed43_test_predictions.csv --out-dir table1_results
```

The script uses the frozen validation-F1 thresholds in `paper_protocols.json`.
It reports average precision, MCC and F1 separately for each seed, then their mean
and sample SD. It does not average probabilities. Application defaults use
validation-MCC thresholds and are not the Table 1 operating points.

## Figure 2: task-specific pocket protocols

`paper_protocols.json` contains the exact parameters.

- Metal: probabilities from seeds 41/42/43 are weighted 0.4/0.2/0.4 before selecting
  ceil(10% of mask-valid residues) per protein. Components use CA distances
  <=8.5 angstrom; centers are probability-weighted CA centroids. Ranking is max
  probability + 0.05 ln(1+size).
- Nucleic acid: ensemble probabilities use the validation-F1 threshold
  0.7542161000000001 and an 8 angstrom component radius. Centers are unweighted
  CA centroids; ranking is mean probability + 0.05 ln(1+size).
- Small molecule: ensemble probabilities select ceil(10% of eligible residues),
  using radius 6 angstrom, probability-weighted CA centers and max probability
  + 0.05 ln(1+size), as for Metal.
- All tasks retain singletons as eligible ranked candidates.
  Ties prefer larger size, then smaller first residue position.
- Reference centers are unweighted annotated-residue CA centroids in the same
  frozen ESMFold structures. Top-N uses the reference site count of each protein;
  every reference independently checks its nearest center among those candidates.
  Hits require strict DCC <10 angstrom. Recall is site-weighted.
- All three tasks have one fixed ensemble recall; there is no seed SD.
  Metal: 127/198 = 64.14%; nucleic acid: 78/107 = 72.90%; small molecule:
  210/336 = 62.50%. The previous Figure 2 protocol is retained in
  `paper_protocols_before_ensemble_freeze.json`.

The application's default pocket code now implements these Figure 2 transformations.
The benchmark uses task masks defining eligible residues. In a new unlabeled
protein all modeled residues are eligible. Replaying benchmark evaluation must
preserve those masks; `formal_candidate_pockets` accepts an eligibility mask.
Full-cohort raw feature/annotation distribution and fresh released-package
inference remain outstanding. Metal parameters and the nucleic-acid 8 angstrom
radius were chosen after prior test exposure; freezing does not remove that
selection limitation.

## Verification scope

New seed41/43 exports preserve every original state-dictionary tensor exactly.
All bundled scaler arrays match the previously reconstructed preprocessing
sidecar. `historical_preprocessing_verification.json` records the prior raw-input
validation replay for all three research checkpoints (tolerance 1e-6). It is a
historical verification, not a new full-cohort release acceptance.
