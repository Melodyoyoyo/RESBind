# RESBind V6 local release candidate

V6, bidirectional contacts, seed 42, no refiner. This directory is staged locally,
not published or ready for unrestricted redistribution. The bundle contains the
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
installed package. MCP integration and Skill setup remain unvalidated.

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
