---
name: resbind
description: Install and run RESBind V6 for protein binding-residue prediction, including compute selection, FASTA checks, ProtT5 and ESMFold preparation, NetSurfP CSV acquisition, and local MCP or CLI inference. Route labeled evaluation and training requests separately.
---

# RESBind

Repository: https://github.com/Melodyoyoyo/RESBind. Prefer a user-supplied local
RESBind checkout or trial ZIP when available; that route needs no GitHub account.
For downloading from the currently private repository, require the user's
legitimate access; never substitute a similarly named model.
This is a development candidate, not a certified turnkey public service.

Preserve V6 seed42, no refiner, bidirectional contacts built from ESMFold C-alpha
coordinates at <=10 Angstrom without self edges. Keep checkpoint, scalers and
validation thresholds together. Do not retrain or alter thresholds for ordinary prediction.

## Route the request

- Prediction: one protein FASTA per job, 20–1024 residues, no labels or data split.
  Read [prediction.md](references/prediction.md) for the complete sequence-to-result workflow.
- Evaluation/training: read [labeled-data.md](references/labeled-data.md) first.
  A FASTA alone is not supervised training data. Do not infer negatives from missing labels.
- MCP setup/calling: read [mcp.md](references/mcp.md). MCP currently executes
  saved-feature CPU prediction; browser/terminal tools handle upstream preparation.

Inspect input before expensive work. A cache hit requires exact normalized
sequence and matching feature/model provenance, not merely an accession match.
It does not establish that a protein is absent from training data.
Report B/Z/U/O/J-to-X substitutions; let the user resolve biologically meaningful
ambiguity before submitting or computing a changed sequence. Do not silently
trim, split, concatenate chains, or discard invalid entries.

For multiple sequences, report their count and lengths, choose a resource-aware
serial or bounded-batch plan with the user, and keep one job and provenance record
per sequence. Do not submit a whole collection to an external service by default.

## Compute and permissions

Before downloading large models or choosing a compute host, read
[compute.md](references/compute.md). It routes saved-feature CPU, local CUDA,
ordinary remote servers (including laboratory-owned machines) and scheduled HPC
use, and defines the handoff when the agent cannot connect.
For HPC, then read [hpc.md](references/hpc.md): a separate HPC-access skill is
optional, not a prerequisite. Discover the user's actual login and scheduler
procedure; never inherit the developer's cluster settings.

Offer existing-feature CPU inference, local CUDA GPU, an authorized laboratory,
personal or rented remote server, or an available HPC compute allocation.
HPC is not mandatory; ownership and execution method are separate decisions.
Inspect resources before selecting a route. Do not promise heavy feature models
will fit a particular GPU or CPU until tested. Never run heavy models on an HPC
login node, start a paid instance, acquire credentials, or upload confidential
sequences without the corresponding user authorization.

Use a dedicated Python environment. Do not change a research environment or
global agent configuration implicitly. Keep personal hostnames/SSH keys out of
shared instructions. A website displaying examples supplies no compute resources.

## Evidence and handoff

Record repository revision, checkpoint checksum, normalized sequence checksum,
upstream model snapshot revisions, NetSurfP source/job, feature checksums,
commands/environment and output locations. Verify output length, residue order,
finite probabilities and successful completion; describe cached versus fresh steps.

Return all three residue tracks and downloadable files. Schema 3 exports all
three tasks' spatial candidates; singleton components are isolated_residue, not
pockets. Report pocket_counts and isolated_residue_counts separately. These
components are not necessarily geometric cavities. Never present predictions as experimental binding
evidence or claim this draft Skill has passed independent full-workflow acceptance.
