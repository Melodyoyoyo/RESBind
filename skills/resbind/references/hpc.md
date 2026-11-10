# GPU cluster workflow

Use the cluster's normal login method, scheduler, and project allocation to generate ProtT5 and ESMFold features.

## Prepare the job

1. Identify the authorized project directory, GPU queue, account, and time limits from the cluster documentation or an existing working job script.
2. Prepare a Python environment with RESBind's feature dependencies and local snapshots of the upstream models.
3. Create a run directory containing the input FASTA and a scheduler script with the selected environment, model paths, and output locations.
4. Run the two feature-generation commands from [prediction.md](prediction.md) sequentially within the GPU allocation.
5. Submit the job and retain its identifier and log path.

Run model computation on allocated compute nodes. Use the cluster's supported transfer and installation locations for large downloads.

## Retrieve outputs

After successful completion, check the sequence and retrieve:

- `features-prott5/prott5.npz`
- `features-prott5/provenance.json`
- `features-esmfold/structure.pdb`
- `features-esmfold/provenance.json`
- The job log

Combine these outputs with the matching NetSurfP CSV and continue with CPU prediction. If a job is interrupted, inspect its state and saved outputs before starting a replacement.

## User-operated submission

When the user operates the cluster connection, prepare the FASTA, job script, environment instructions, submission command, and return-file checklist for their usual workflow. Resolve scheduler and path settings before submission. Resume prediction after the returned features pass input validation.
