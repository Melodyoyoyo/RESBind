# First-time HPC access, with or without a site skill

An HPC-access skill is optional. If the user provides one, read it and its required
site documentation; otherwise discover the same facts from the user's normal
procedure and official cluster documentation. Never reuse the author's hostname,
account, SSH key, partition, paths or resource request on another user's cluster.

## Establish the access contract

Start with one practical question rather than a long technical questionnaire:

> How do you normally log in: SSH/VS Code, a terminal application, or your
> institution's web portal? You can share the normal command or cluster guide
> with passwords/tokens removed. You do not need to write another Skill.

With permission, inspect the named SSH configuration entry (not private key
contents) or the user's supplied job script. Resolve only relevant facts:

- Connection method/alias, any VPN or jump host, and whether interactive MFA is
  required. Do not alter network/security settings to force access.
- User-authorized project/scratch directory and installation/download policy.
- Scheduler or permitted allocation method; allowed GPU queue/partition, required
  account/project/QoS, resource limits and wall time.
- Available Python/modules/containers and model storage; reuse only compatible,
  authorized resources. Do not assume personal quota permits large weights.

Ask the user or consult the site's own instructions for unknown essentials. A
failed `sbatch` lookup is not permission to run GPU models on the login node.
Do not guess billing accounts, queue names or memory requests. Private-repository
access on the local host does not imply access from the cluster.

## Decide who operates the connection

**Agent can use authorized SSH/terminal:** connect using the agreed method, enter
the agreed workspace and perform lightweight checks. Let the user complete any
password/MFA step in the normal login interface. Do not request secrets in chat,
save them in scripts or repeatedly retry failed authentication. Pause for user
action if VPN/MFA/network policy prevents access; no security bypasses.

**User can connect, agent cannot:** do not demand a new HPC Skill. Prepare a
handoff package for their existing portal/SCP/SFTP workflow. The user performs
login/upload/submission; the agent resumes from logs and returned artifacts.
If the scheduler details are unknown, first ask for a redacted working GPU-job
example or the cluster guide, not for the user to write a new program.

## Prepare and execute a site-specific job

1. Create a new run directory only within the agreed workspace. Record FASTA
   hash and the exact RESBind revision. For an offline/private-repo cluster,
   transfer a verified code archive rather than embedded Git credentials.
2. Check scheduler availability and applicable quota/limits using lightweight
   commands from that site's guide. Do not scan unrelated users' files/jobs.
   For Slurm, relevant commands include `sinfo`, the user's `squeue`, and `sacct`;
   use the site's equivalents for PBS/LSF or an approved interactive allocation.
3. Resolve environment and snapshot paths. An existing environment is read-only
   unless changes are authorized; otherwise create an isolated environment under
   the allowed project path. Weight downloads/installations must use a permitted
   transfer/setup location. Do heavy model loading and computation only within
   a compute allocation, never on the login node.
4. Generate a job script using confirmed scheduler directives, actual paths,
   resource limits and log locations. Quote paths, fail on command errors, use
   task-local caches/temp files, and run prediction.md's two feature commands
   sequentially in distinct processes with fresh output directories. Review the
   generated script for unresolved fields before submission. Do not copy a fixed
   A6000/Slurm request as a universal template or claim other schedulers tested.
5. Submit once and immediately record the returned job ID. If submission output
   is interrupted or ambiguous, inspect the user's queue before any resubmission.
   Poll that job at reasonable intervals; a long pending state is not failure.
6. Inspect exit status and logs, then verify sequence-bound NPZ/PDB and provenance.
   Job completion alone is insufficient. Preserve logs for failed stages and
   reuse only validated successful stages. If partial output exists, choose a new
   output directory for a retry; the launcher intentionally refuses overwrite.
7. Transfer the validated outputs and checksums for CPU inference. Do not cancel
   other jobs, erase research caches or change shared environments for cleanup.

## Human handoff package

After site facts are known, produce in an agreed local folder:

- Checked FASTA, exact code archive or accessible revision, and checksums.
- A runnable site-specific feature job script and environment/setup instructions
  permitted by that site; do not include secrets or unlicensed redistributed tools.
- Exact upload destination and submission command for the user's normal tool,
  plus how to obtain the job ID, status and logs. All host/path/resource fields
  must be resolved before describing the script as ready to run.
- Return-file checklist: `features-prott5/prott5.npz`,
  `features-prott5/provenance.json`, `features-esmfold/structure.pdb`,
  `features-esmfold/provenance.json`, job log and exit status. State if an already
  supplied feature means only one stage is needed.

End the handoff with the exact next human action and resume point. After the user
returns outputs, verify sequence/hashes and continue the same run using the
documented MCP/CLI. Do not require them to understand model code. Label the route
human-assisted, not unattended automation. If no authorized compute or required
site information is available, preserve the preparation and report that specific
blocker rather than silently falling back to unsupported CPU feature generation.
