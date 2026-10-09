"""Model-free input routing and explicit resource inspection."""
from __future__ import annotations

import hashlib
import platform
from pathlib import Path

from .fasta import parse_single_fasta


def workspace_file(root: Path, value: str, max_bytes: int) -> Path:
    root = root.resolve(strict=True)
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Input must stay inside RESBIND_MCP_WORKSPACE.")
    if not path.is_file() or path.stat().st_size > max_bytes:
        raise ValueError("Input is missing, not a file, or exceeds the size limit.")
    return path


def inspect_input(fasta_text: str, mode: str = "predict", *, pdb=None,
                  netsurfp=None, prott5=None) -> dict:
    if mode not in {"predict", "evaluate", "train"}:
        raise ValueError("Mode must be predict, evaluate, or train.")
    lines = [line.strip() for line in fasta_text.splitlines() if line.strip()]
    if not lines or not lines[0].startswith(">"):
        raise ValueError("Provide FASTA beginning with a nonempty >identifier header.")
    if sum(line.startswith(">") for line in lines) != 1 or not lines[0][1:].strip():
        raise ValueError("Provide exactly one nonempty FASTA header per job.")
    raw = "".join("".join(line.split()) for line in lines[1:]).upper()
    if raw and set(raw) <= set("ACGTUN"):
        raise ValueError("Input appears to be nucleotide rather than protein sequence.")
    protein = parse_single_fasta(fasta_text)
    missing = []
    if pdb is None:
        missing.append("structure")
    else:
        from .structure import parse_pdb_ca, validate_structure_sequence
        validate_structure_sequence(parse_pdb_ca(Path(pdb)), protein.sequence)
    if netsurfp is None:
        missing.append("netsurfp")
    else:
        from .features import load_netsurfp_table
        load_netsurfp_table(Path(netsurfp), protein.protein_id, protein.sequence)
    if prott5 is None:
        missing.append("prott5")
    else:
        from .features import load_prott5_npz
        load_prott5_npz(Path(prott5), protein.sequence)
    return {
        "mode": mode, "protein_id": protein.protein_id, "length": len(protein.sequence),
        "normalized_sequence": protein.sequence,
        "sequence_sha256": hashlib.sha256(protein.sequence.encode()).hexdigest(),
        "normalizations": [{"position": i+1, "from": aa, "to": "X"}
                           for i, aa in enumerate(raw) if aa in "BZUOJ"],
        "missing_inputs": missing, "labels_required": mode != "predict",
        "ready_for_saved_prediction": mode == "predict" and not missing,
        "ready_for_supervised_execution": False,
        "supervised_note": "Evaluation/training require explicit labels, missing-label masks and a validated split policy; no automatic split is performed.",
        "novelty_note": "Compatible cached features do not establish training-set membership or novelty.",
    }


def inspect_compute() -> dict:
    import torch
    available = torch.cuda.is_available()
    return {"platform": platform.system(), "python": platform.python_version(),
            "torch": torch.__version__, "cuda_available": available,
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if available else [],
            "scope": "Saved-feature V6 CPU inference is supported. Heavy feature-model hardware minima are not yet certified; no model was loaded."}
