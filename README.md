# Current pocket release: v0.2.2

All three tasks independently evaluate seeds41/42/43 and report metric mean +/- sample SD (ddof=1). Probabilities and candidate sets are not averaged or pooled. Strict DCC<10 A Top-N: Metal 62.79 +/-2.87%, nucleic acid 72.27 +/-2.16%, small molecule 64.58 +/-0.79%. Metal retains top10%, radius8.5, weighted centers and max-score-plus-size; other tasks retain validation-F1 thresholds, original radii, unweighted centers and mean-score-plus-size. Checkpoints and Table1 unchanged. The default bundle directory and predictor class retain legacy ensemble names only for path/API compatibility.

# RESBind

Protein binding-residue prediction for metal, nucleic acid and small molecule
tasks. This research release includes three Full checkpoints (seeds 41/42/43),
with bidirectional contacts and no refiner. Pocket prediction defaults to the
three independent models for all pocket tasks; residue operating-point tracks retain seed42,
with all three seed probability tracks also exported.
The bundle contains the
unchanged trained checkpoint, four scalers and fixed validation-MCC thresholds.
Input contacts are built from ESMFold C-alpha coordinates at <=10 Angstrom,
without self edges; pooling uses both directions. Pocket clustering is separate.

For Table 1 evaluation and task-specific Figure 2 parameters, see
[paper reproducibility](reproducibility/README.md). Table 1 summarizes independently
evaluated seeds; the current pocket protocol likewise evaluates all tasks independently per seed.

## Get the code and model

The repository can be downloaded without a GitHub account:

```sh
git clone https://github.com/Melodyoyoyo/RESBind.git
cd RESBind
python -m venv .venv
```

Activate the environment with `.venv\Scripts\activate` on Windows Command Prompt,
or `source .venv/bin/activate` on Linux/macOS. Alternatively, select its Python
executable directly in your agent. Python 3.12 is the tested CPU configuration;
this is not a guarantee for all operating systems. Without Git, use **Code >
Download ZIP**, extract it, and open the extracted directory before creating the
environment. Keep the bundled model directory alongside the code.

For agent-assisted setup, start with [the Skill](skills/resbind/SKILL.md).
MCP is optional; it provides local tools, not a hosted prediction service.

## Archived-input smoke test

If you received a repository or trial ZIP, extract it into a new folder and use the
RESBind directory inside it as the repository root. No GitHub account or Git
clone is required. Start with skills/resbind/SKILL.md; the separate Skill ZIP is
instructions only. The trial includes weights and P07014 inputs, not Python
dependencies or the large ProtT5/ESMFold upstream models. Dependency installation
may require internet. This is not a completely offline distribution.

Install Python >=3.9 and an appropriate PyTorch distribution. From this directory:

```sh
python -m pip install .
resbind predict --fasta examples/P07014/sequence.fasta --pdb examples/P07014/structure.pdb --netsurfp-csv examples/P07014/netsurfp.csv --prott5-npz examples/P07014/prott5.npz --device cpu --out-dir results/P07014
```

The default model path is models/v6_bidirectional_ensemble relative to the current
directory. From another directory set RESBIND_BUNDLE_DIR to the absolute bundle
directory. RESBIND_JOB_ROOT and RESBIND_CACHE_ROOT select writable runtime paths.
The Python wheel does not embed model weights; retain the models directory.
This example uses saved ESMFold, ProtT5 and NetSurfP outputs, not newly generated
features. Inputs must match FASTA. Supplying a PDB does not prove it came from
ESMFold; this example's provenance is preserved in release_manifest.json.

## Clean environment validation (2026-10-09)

A fresh Windows venv using Python 3.12.7, with system-site-packages disabled,
successfully installed this package and its dependencies via normal pip resolution.
Before installation, numpy, pandas, torch and resbind_server were absent. After
installation, their import paths were all inside the new venv. No dependencies
were borrowed from the research environment and no research module was imported.

Tested versions include torch 2.14.1, numpy 2.5.3 and pandas 3.0.6. The saved-input
P07014 CPU run matched all three task calls; maximum absolute score difference
from the archived reference was 2e-7 (tolerance 1e-6). pip check, the CLI entry
point, result ZIP contents and cached rerun passed. See
`clean_install_verification.json` and `requirements-tested-windows-py312.txt`.

To select the recorded dependency versions on this Windows/Python combination:

```sh
python -m pip install -c requirements-tested-windows-py312.txt .
```

This establishes clean-venv installation on the existing Windows machine, not
validation of a new physical machine, Linux/macOS, GPU execution, training or
new-sequence feature generation. The earlier `verification.json` separately
records code-isolation testing with existing research dependencies.

## New-sequence feature generation

The optional features dependencies are needed for ProtT5/ESMFold generation.
These large models need appropriate compute; hardware minima and installation
of the optional feature-generation dependencies have not been validated in a
clean environment. NetSurfP CSV import is supported.
No NetSurfP software, credentials or automated cloud provider is distributed.
P62942 has now passed one new-sequence engineering acceptance in the research
pipeline: fresh HPC ProtT5/ESMFold plus a CSV downloaded by the agent from the
official DTU web form (job 6AC8CAC2001551F359F9C19B, 108 residues). This does not
establish a robust automated feature provider or a new-sequence test of this
installed package. The newer local MCP test is described below; independent
full-workflow Skill acceptance remains outstanding.

## Agent Skill and local MCP

The full-workflow instructions are in [skills/resbind/SKILL.md](skills/resbind/SKILL.md).
They cover installation, input validation, resource selection, NetSurfP browser
acquisition, ProtT5/ESMFold preparation, prediction and labeled-data routing.
No GitHub account is required to download this public repository.

Compute selection is documented in [the resource guide](skills/resbind/references/compute.md):
saved-feature CPU, local CUDA, ordinary remote servers (laboratory-owned, personal
or rented), and scheduler-managed HPC. Ownership does not determine the execution route. The
[HPC onboarding guide](skills/resbind/references/hpc.md) supports users with no
separate access skill, including human login/submission when agent SSH is unavailable.
These are operating instructions, not evidence that every hardware/site was tested.

For the local stdio bridge use Python >=3.10 (tested with 3.12):

```sh
python -m pip install ".[mcp]"
```

Configure `RESBIND_MCP_WORKSPACE` as a dedicated existing input/output directory,
and `RESBIND_BUNDLE_DIR` as the absolute trusted model bundle path. Launch the
environment's Python with `-I -m resbind_server.mcp_server` from the agent client.
Tools: `check_environment`, `inspect_fasta`, `predict_saved`.
The prediction tool requires matching FASTA, PDB, NetSurfP CSV and ProtT5 NPZ;
it runs on CPU and returns local output paths. No HTTP listener, GPU model download,
remote job submission, training or persistent queue is provided by this bridge.
Use an owner-controlled workspace; path checks are not an OS security sandbox.

Run the protocol acceptance test after installation:

```sh
python tests/mcp_acceptance.py .
python -m unittest discover -s tests -v
```

The `python -m resbind_server.feature_stage` launcher generates one upstream
feature type per separate CUDA process using a complete local model snapshot.
It passed a P62942 GPU acceptance on 2026-10-09: GitHub revision 38abe57 was
installed into a new target directory, then fresh ProtT5 and ESMFold outputs
were generated on an RTX A6000 (Slurm job 379654, 4m55s total). The existing
HPC dependency environment was reused: Python 3.10.18, torch 2.6.0+cu124,
transformers 5.3.0. This is not a clean GPU dependency-installation test.

Those newly generated features, together with the previously obtained and
hash-verified DTU NetSurfP CSV, passed local MCP V6 inference. All 108 residues,
three score tracks, thresholds, feature checksums and ZIP contents were checked.
All residue calls matched the previous research run; maximum score difference
was approximately 1e-7. See `release_gpu_verification.json`. Model-loader warnings
were retained, not suppressed by altering weights/configuration. This one-case
engineering check is not biological validation or a guarantee for every GPU/length.

See the Skill for invocation and provenance requirements. Existing verification
JSON files describe earlier core tests; `mcp_verification.json` records the
separate saved-input MCP protocol run. Independent complete Skill replay,
fresh feature-model download/install and general hardware support remain open.

On 2026-10-09 the installed package passed the official SDK 1.30.0 stdio test:
all three tools were discovered/called, invalid paths rejected, and the P07014
CPU prediction matched all three task calls (maximum score difference 2e-7).
No actual agent-client configuration or independent new-protein Skill replay
is implied by this protocol test.

Keep the legacy `web` extra in a separate environment: its Pydantic constraint
conflicts with the MCP SDK dependency. The static showcase needs neither extra.

### Independent saved-feature Skill trial

A fresh-context agent tested revision 2f3ce3a without prior conversation history
or implementation hints. It cloned the private repository, checked all 48 recorded
file hashes, installed into a new Windows venv (no system-site packages), read the
Skill and invoked the three tools through the official MCP stdio client. P62942's
108 residues, all three tracks, task-wise candidates and ZIP integrity passed;
no product code changes were required. See independent_skill_verification.json.

This is independent **supplied-feature CPU** acceptance, not a complete fresh
upstream-generation trial or persistent desktop-client setup. Authorized execution
outside the test sandbox was needed for installation and MCP startup. CLI progress
currently uses generic "Generating" labels even for imported features; consult
summary.input_sources and checksums to distinguish imported versus generated data.

## Training

training/train_v6.py retains the research training implementation, with explicit
input/output paths required and bidirectional contacts enabled by default:

```sh
python -m pip install ".[training]"
python training/train_v6.py --feature-root YOUR_FEATURES --embedding-dir YOUR_EMBEDDINGS --contact-dir YOUR_ESMFOLD_CONTACTS --out-dir YOUR_OUTPUT --bidirectional-contacts --loss-balancing none --seed 42
```

Use the original split-specific feature tables and matching cached embeddings and
ESMFold contacts. Training data preparation/distribution and reproduction of the
full training run are not yet validated in this candidate. No refiner is included.

## Scope and release gates

Three tasks: metal, nucleic acid, small molecule. Pipeline schema 3 exports all
three tasks' spatial components in pockets.csv, with task, candidate_type,
threshold and cutoff columns. Singletons are labeled isolated_residue and excluded
from pocket_counts; they remain in the CSV and summary's pockets_by_task.
Residue lists use 1-based sequence positions, not arbitrary PDB residue numbers.
All tasks retain independent seed probabilities. Metal retains 8.5 Angstrom clustering, top10% selection, weighted centers and max probability plus a size term. Other tasks use independent validation-F1 thresholds, ordinary centroids and mean probability plus the size term: nuclear radii 12/8/8 Angstrom and small radii 6/6/6. The default visualization uses seed42 for all tasks; all per-seed candidates are exported in the ZIP. See the paper protocol JSON.
These candidate regions are not necessarily geometric cavities.

Historically, all 15 website cases x 3 tasks matched the stored website outputs at tolerance
1e-10 when given identical scores/structures; see pocket_parity_verification.json.
Model contact construction remains bidirectional 10 Angstrom. Cache schema was
bumped from 2 to 3, so old small-only outputs are not reused. The previous
n_small_molecule_pockets summary field is replaced by task-wise pocket_counts
and isolated_residue_counts; downstream readers must account for this schema change.
Historical acceptance reports retain the configuration and limitations of their
recorded revision; they do not describe the current independent-seed protocol. The new candidate-code
parity and supplied-feature runtime checks are in `reproducibility`.
Predictions are not experimental validation. The optional Gradio interface is a local prototype.
Remaining reproducibility work includes clean optional-feature dependency
installation and complete training-data/split preparation documentation. This
release is not a complete reproduction package for every manuscript experiment.
No plotting scripts, website, private server settings or research caches ship.

## License

Original RESBind code, documentation and the V6 checkpoint/scalers are provided
under the [MIT License](LICENSE), copyright 2026 Minghan Chen. Third-party
resources are not relicensed; see [license scope and third-party notices](THIRD_PARTY_NOTICES.md).
No provider endorsement or special permission for third-party outputs is claimed.
A software license is not evidence of biological validity or a guarantee of
model performance.
