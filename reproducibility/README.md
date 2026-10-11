# Paper reproduction

The default three-checkpoint bundle preserves the frozen metal probability ensemble (weights 0.4/0.2/0.4). Nucleic-acid and small-molecule pocket predictions have been restored to independent-seed evaluation. No refiner is used. The exact selectors, radii, centers, ranking, reference/matching rules and Table1 thresholds are in paper_protocols.json.

Table1 remains independent seed41/42/43 mean and sample SD. `prob_*` retains seed42 residue outputs, while `prob_*_seed41/42/43` exports all probabilities. `pocket_prob_metal` is weighted; other `pocket_prob_*` columns retain seed42. The default pockets.csv and UI show seed42 for nucleic-acid/small-molecule tasks. The result ZIP additionally includes pockets_independent_seeds.csv and summary.json has each seed's candidates separately. Mean recall is computed after separate evaluation, never by merging candidate sets.

Strict DCC <10 Angstrom Top-N: metal 64.14%; nuclear 72.27 +/-2.16%; small molecule 64.58 +/-0.79%. Metal uses top10%, radius8.5, probability-weighted centers, max-score-plus-size. Nuclear uses original seed thresholds and radii12/8/8; small molecule original seed thresholds and radius6 for every seed. Both use unweighted centers and mean-score-plus-size. Singletons remain eligible candidates. Benchmark replay requires the original cohort, mask-valid residues, annotations and coordinate mapping.

restored_pocket_verification.json checks full-cohort candidate parity and DCC using archived probabilities. It does not certify fresh full-cohort feature generation or inference. Historical frozen_pocket_code_verification.json and ensemble_runtime_verification.json describe the superseded all-task ensemble. The restored CPU pipeline was also run on supplied P07014 features; all 11 existing unit tests passed.

Metal retains its recorded test-exposure limitation. Restoring old methods does not create a new untouched test set. Checkpoint and scaler bytes are unchanged.
