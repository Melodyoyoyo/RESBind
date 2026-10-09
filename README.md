# RESBind V6 local release candidate

V6, bidirectional contacts, seed 42, no refiner. This is a private development
candidate, not ready for unrestricted redistribution. The bundle contains the
unchanged trained checkpoint, four scalers and fixed validation-MCC thresholds.
Input contacts are built from ESMFold C-alpha coordinates at <=10 Angstrom,
without self edges; pooling uses both directions. Pocket clustering is separate.

## Archived-input smoke test

Install Python >=3.9 and an appropriate PyTorch distribution. From this directory:

```sh
python -m pip install .
resbind predict --fasta examples/P07014/sequence.fasta --pdb examples/P07014/structure.pdb --netsurfp-csv examples/P07014/netsurfp.csv --prott5-npz examples/P07014/prott5.npz --device cpu --out-dir results/P07014
```

The default model path is models/v6_bidirectional_seed42 relative to the current
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
The repository is private: other users need authorized access until public release.

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

Three tasks: metal, nucleic acid, small molecule. The existing CLI exports only
legacy small-molecule candidate pockets (6 Angstrom components); it does not yet
match the static website's all-task pocket postprocessing. Predictions are not
experimental validation. The optional Gradio interface is a local prototype.
Before public release: verify optional dependency installation, data/model redistribution
rights, choose a license, document training inputs, reconcile pocket outputs,
and complete new-input acceptance. No license grant is asserted by this staging.
No plotting scripts, website, private server settings or research caches ship.
