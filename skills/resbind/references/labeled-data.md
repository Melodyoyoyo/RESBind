# Evaluation and training routing

These modes require additional dataset validation and are not turnkey MCP tools.
Before execution, ask whether the user wants frozen-model evaluation, exact
training reproduction, fine-tuning, or new training. Do not treat these as synonyms.

Require per-protein sequence and per-residue labels for the relevant tasks, their
provenance, numbering/chain mapping, and an explicit missing-label mask. Unknown
or unannotated does not mean experimentally negative. Check length and index
alignment before building dataset tables. FASTA headers are not a label schema.

For frozen evaluation, preserve the trained checkpoint and validation thresholds;
report overlap with known training/validation data when that information exists.
Do not optimize thresholds on the test labels. If training membership is unknown,
state that rather than calling a new accession an independent test sample.

For original-run reproduction, retain the original train/validation/test membership
and preprocessing fit scope. For a new dataset, agree a protein/homology-aware split
policy, family/identity criterion and seed before generating splits. Do not randomly
split residues or silently replace user-provided splits. Fit scalers only on training
data; select thresholds/model using validation, then evaluate once on held-out test.

The repository includes `training/train_v6.py`; inspect `--help` and its required
split-specific table/embedding/contact schema before adapting data. Training input
packaging and reproduction are not validated by the CPU inference smoke test.
Report a concrete missing-input/schema issue instead of fabricating labels, splits,
feature tables or claiming an unexecuted training run succeeded.
