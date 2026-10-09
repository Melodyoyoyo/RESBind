"""One upstream model per process, using an explicitly selected local snapshot."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path

from .workflow import inspect_input


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prott5", "esmfold"], required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    info = inspect_input(args.fasta.read_text(encoding="utf-8"))
    if info["normalizations"]:
        raise ValueError("Normalize the FASTA explicitly after reviewing reported substitutions before generating features.")
    model_dir = args.model_dir.resolve(strict=True)
    if not (model_dir / "config.json").is_file():
        raise ValueError("Provide a complete local model snapshot containing config.json.")
    import torch
    if not torch.cuda.is_available():
        raise ValueError("This feature-stage launcher requires CUDA; heavy-model CPU execution is unvalidated.")
    if args.out_dir.exists():
        raise ValueError("Choose a new output directory; existing output is never overwritten.")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    from .features import ProtT5Provider, ESMFoldProvider, load_prott5_npz
    from .structure import parse_pdb_ca, validate_structure_sequence
    import numpy as np
    args.out_dir.mkdir(parents=True, exist_ok=False)
    sequence = info["normalized_sequence"]
    if args.stage == "prott5":
        provider = ProtT5Provider("cuda")
        provider.model_name = str(model_dir)
        embedding = provider.embed(sequence)
        output = args.out_dir / "prott5.npz"
        with output.open("xb") as handle:
            np.savez_compressed(handle, sequence=np.array(sequence), embedding=embedding)
        load_prott5_npz(output, sequence)
    else:
        provider = ESMFoldProvider("cuda", chunk_size=64)
        provider.model_name = str(model_dir)
        output = args.out_dir / "structure.pdb"
        with output.open("x", encoding="utf-8") as handle:
            handle.write(provider.fold(sequence))
        residues = parse_pdb_ca(output)
        validate_structure_sequence(residues, sequence)
        if not np.isfinite([[r.x, r.y, r.z] for r in residues]).all():
            raise ValueError("Structure contains non-finite coordinates.")
    provenance = {"stage": args.stage, "sequence_sha256": info["sequence_sha256"],
                  "model_directory": str(model_dir), "torch": torch.__version__,
                  "model_config_sha256": hashlib.sha256((model_dir / "config.json").read_bytes()).hexdigest(),
                  "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    (args.out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
