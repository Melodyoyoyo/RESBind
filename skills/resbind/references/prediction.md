# Complete prediction workflow

## 1. Obtain and verify RESBind

Clone https://github.com/Melodyoyoyo/RESBind using the user's authorized Git access,
record `git rev-parse HEAD`, and read that revision's README. Private-repository
access failure is an access issue, not a reason to request a password in chat.
Check files against `release_manifest.json` before installing or running weights.
Create a dedicated environment: Python 3.12 was tested; MCP requires >=3.10.
From the checkout run `python -m pip install ".[mcp]"` for local MCP, or
`python -m pip install .` for CLI only. Run `python -m pip check`.
The optional legacy Gradio `web` extra has a conflicting Pydantic pin; do not
combine it with `mcp` in the same environment. The static website does not need it.

First run the README's saved-input P07014 CPU example. This checks installation
and weights without starting ProtT5, ESMFold or submitting a NetSurfP job.

## 2. Validate and select compute

Read [compute.md](compute.md) to select a route. If HPC is selected, also read
[hpc.md](hpc.md), even when the user has no cluster-access skill. Establish where
the agent can run commands versus where a human must log in or submit a job.

Use MCP `inspect_fasta` or Python `resbind_server.workflow.inspect_input`.
Accept a single nonempty FASTA record; check lengths and reported substitutions.
For provided features, validate their contents against this exact sequence.
Check resource availability using MCP `check_environment` or a terminal GPU/memory
inspection. This does not certify sufficient capacity for feature generation.
MCP reports only the environment of its own server process: it cannot discover
an SSH host's GPU or an HPC allocation. Check those on the selected machine.

Reusing user-provided ProtT5, NetSurfP and ESMFold outputs is valid when their
sequence and provenance match. Do not silently substitute an experimental PDB,
AlphaFold structure, different embedding model or alternate surface predictor.

## 3. Generate missing ProtT5 / ESMFold features

Do this only on the selected CUDA compute machine/allocation, in a feature-specific
environment if dependencies differ from the CPU/MCP environment. Install the
package's `features` extra and run import checks before reserving long compute time.
The launcher passed one 108-residue P62942 acceptance on RTX A6000 with an existing
Python 3.10.18 / torch 2.6.0+cu124 / transformers 5.3.0 environment and local weights.
That is not clean-environment installation or validation on all GPUs/lengths.
See `release_gpu_verification.json` in the repository for the tested scope.

Obtain complete snapshots from the official upstream repositories
`Rostlab/prot_t5_xl_uniref50` and `facebook/esmfold_v1`. Read their model cards and
terms, resolve and record concrete revisions, check free disk space, and obtain
authorization for the download. Use the upstream supported download tooling;
do not execute downloaded custom code or replace model variants implicitly.

Run each stage as a separate terminal process so its GPU memory is released before
the next stage. In the following commands, `input.fasta` is the checked input;
`prott5-model` and `esmfold-model` are the downloaded local snapshot directories,
and both output directories must be new:

```sh
python -m resbind_server.feature_stage --stage prott5 --fasta input.fasta --model-dir prott5-model --out-dir features-prott5
python -m resbind_server.feature_stage --stage esmfold --fasta input.fasta --model-dir esmfold-model --out-dir features-esmfold
```

Launcher runs offline using local weights, requires CUDA, writes sequence-bound
NPZ/PDB and provenance, and refuses existing output directories. Preserve failure
logs; inspect an OOM before retrying, rather than repeatedly increasing workloads.
Check 1024 embedding columns, finite values, structure sequence and feature hashes.
For remote compute, transfer only authorized input/output files and verify hashes.

## 4. Obtain missing NetSurfP CSV

Use https://services.healthtech.dtu.dk/services/NetSurfP-3.0/ in the agent's browser.
Explain that submitting sends the sequence to DTU and establish authorization.
Use the normal official FASTA form; this browser operation is agent automation.
Record the job URL/ID as soon as it is available, before waiting. Follow current
service terms, capacity restrictions and displayed status. Do not invent a hidden API.

On completion, use Export All / CSV (or the currently displayed equivalent),
download the actual file, and keep its SHA256 and job metadata. Validate required
columns, complete unique residue numbering, sequence, finite values and ranges
with RESBind's loader. A results page alone is not a downloaded usable CSV.

On a slow job, inspect the recorded job rather than resubmitting duplicates.
Report the last observed state and resume via that URL. Stop for user-required
authentication/CAPTCHA/terms or service errors instead of bypassing controls.
If browser access is unavailable, ask the user to provide the CSV and resume
downstream; do not claim the agent completed the upstream stage.
Do not redistribute NetSurfP software or claim DTU has endorsed this integration.

## 5. Run V6 and verify

With all files local, call MCP `inspect_fasta`, then `predict_saved`. Alternatively,
from the checkout, with the actual prepared file paths:

```sh
resbind predict --fasta input.fasta --pdb features-esmfold/structure.pdb --netsurfp-csv netsurfp.csv --prott5-npz features-prott5/prott5.npz --device cpu --out-dir results
```

Set `RESBIND_BUNDLE_DIR` to the absolute bundle directory if running elsewhere.
Choose fresh writable job/output directories for an acceptance run. Contact maps
and surface summaries are constructed inside the pipeline; do not manually use
the pocket-clustering cutoff to build contacts.

Check summary.json, residue_predictions.csv, pockets.csv, structure.pdb and ZIP.
Three residue tasks and all-task spatial candidates are returned. pockets.csv
includes task and candidate_type; only multi-residue components count as pockets.
Summary metadata records the 6/8/6 Angstrom cutoffs, ranking and centroid rule.
No labels are required and no train/validation/test split is generated.
