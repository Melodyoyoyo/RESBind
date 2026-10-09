# Select compute and an execution route

Read before large downloads or feature generation. No HPC-access skill is required.
This document is a routing procedure, not a guarantee that every route is tested.

## First decision: what is missing?

Validate FASTA and any supplied features using prediction.md. Reuse only exact
sequence-matched, provenance-checked features. If all three features exist, prefer
CPU V6 inference; do not start a GPU merely because one is available. If only
NetSurfP is missing, obtaining its CSV does not require a user-owned GPU.

If ProtT5 or ESMFold features are missing, ask only what is not already known:

> Do you want to use this computer or a remote machine you can access (a laboratory
> server, rented server or HPC)? Do you normally run commands directly there or
> submit jobs to a queue? If unsure, may I check?

Classify by execution method: local, authorized direct remote execution, or
scheduler-managed execution. Record ownership (personal/laboratory/rented)
separately for permissions and costs. A machine called a server is not necessarily
HPC, GPU-equipped, rented or dedicated to this user.

| Available resources | Route | What not to promise |
|---|---|---|
| Complete matching features + CPU | Install core/MCP; run saved-feature inference | No heavy upstream generation is needed |
| Only CPU + missing ProtT5/ESMFold | Offer GPU access or user-supplied matching features; pause missing stages | Current feature launcher requires CUDA; CPU-only FASTA-to-result is not supported |
| Local NVIDIA CUDA GPU | Check resources and isolated feature environment; run upstream stages sequentially | An available GPU does not establish sufficient memory |
| HPC account | Follow hpc.md; use an authorized compute allocation | Do not assume Slurm, a known partition, or unattended SSH |
| Laboratory/personal/rented remote server with CUDA and direct execution permitted | Check access, resource-sharing rules and any cost scope; run remotely | Do not assume exclusive GPU use or buy/start/resize instances implicitly |
| Apple/AMD GPU or uncertain backend | Explain current CUDA requirement; offer other compute or supplied features | Do not silently switch to MPS/ROCm or claim support |

Do not substitute other structures/embeddings/surface predictors, split a protein
or change thresholds to force an unsupported resource route to complete.

## Local CUDA GPU

Inspect on the actual feature-generation host: OS, selected Python environment,
GPU model/free VRAM, driver/PyTorch CUDA availability, available RAM and writable
disk. Use terminal diagnostics such as `nvidia-smi` when available; its absence
alone does not prove the computer has no GPU. Record missing or uncertain facts.
MCP check_environment describes its own process, not all environments on a host.

Use an isolated environment for optional feature dependencies; do not replace a
working research environment. Verify imports and CUDA in that environment. For
Windows, macOS or another untested setup, do not infer support from the tested
Linux HPC run. Propose a compatible environment with the user if installation
fails; changing OS/global drivers is not an implicit implementation step.

Check free disk before upstream-weight downloads. Prefer complete, verified
existing official snapshots; otherwise follow prediction.md for authorized
downloads. Run ProtT5 then ESMFold in separate processes, one sequence at a time
initially. GPU capacity depends on sequence length and software settings; the
tested RTX A6000 case is not a published minimum-memory requirement.

On OOM or model-load failure, retain the log and completed outputs. Diagnose free
VRAM, RAM, sequence length and environment; do not retry indefinitely, kill other
processes, modify weights or silently truncate the sequence. Propose a bounded
retry only after a concrete change, or request different compute.

## Ordinary remote server: laboratory, personal or rented

Ask for the normal connection method, allowed directory, GPU allocation and
whether direct execution is permitted or a scheduler is required. Scheduled
systems follow hpc.md regardless of ownership. Shared laboratory servers may
permit direct execution but require reserving a GPU or agreeing resource limits
with an administrator. Follow those rules; free VRAM alone is not authorization
to occupy a shared GPU. Do not terminate another user's process to make room.

No separate server-access skill is necessary. Use the user's normal connection
procedure and authorized configuration. Perform the local-CUDA environment checks
on the remote machine, not on the agent's laptop. A CPU-only remote server uses
the saved-feature CPU route; its server label does not change requirements.
For permitted direct execution, create an isolated case directory and use a
site-approved persistent session or batch process with a recorded PID and log.
Do not assume an SSH disconnect leaves foreground computation running. Inspect
the recorded process/log after reconnection before starting a duplicate run.

Confirm permission and resource/time limits for the run. For paid machines,
clarify the cost scope; if cost is unknown, do not promise a price or a free run.
Ask before provisioning, resizing or extending paid resources. Account
login/password/MFA belongs in the user's login interface, never in SKILL.md.
Use existing SSH aliases/keys with permission; do not collect private keys.

If the machine cannot reach GitHub/model sources, prepare authorized transfers
from an accessible host. Do not transfer repository credentials, whole home
directories or restricted third-party software. Check revision and file hashes.
Starting/stopping/deleting cloud instances or disks is separate from finishing a
process; follow only an explicit lifecycle instruction after verifying outputs.

## Shared resume and result contract

Keep a local-only run record: selected route, code revision, sequence hash, stage
status, actual output paths, feature hashes, model snapshot revisions, job/process
ID and log path. Keep site-specific connection details out of shared release files.
Use states such as prepared, submitted, running, failed and validated; a queued
job or downloaded page is not a completed stage.

For remote generation, retrieve only the case's ProtT5 NPZ, ESMFold PDB, their
provenance JSON and logs; verify hashes and sequence locally. Keep NetSurfP CSV
provenance too. Then inspect_fasta and predict_saved can run locally on CPU.
Alternatively an authorized remote CPU execution can use the documented CLI;
current local MCP is not a remote job scheduler and needs no public network port.

With no terminal/SSH access, generate a site-specific handoff as described in
hpc.md, or equivalent dedicated-server commands. Explain the next human action,
required return files and where the agent resumes. Do not report an unexecuted
handoff package as a successful end-to-end run.

## Evidence boundary

Repository reports establish saved-feature CPU/MCP tests and limited Linux CUDA
feature-generation tests using existing upstream weights/dependencies. They do
not establish clean GPU dependency installation, all consumer GPU sizes, arbitrary
HPC schedulers or rental providers. This routing guidance adds no new hardware
acceptance claim. Report the current user's actual tested route separately.
