from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class ProteinInput:
    protein_id: str
    sequence: str


@dataclass(frozen=True)
class StructureResidue:
    sequence_position: int
    chain: str
    residue_number: int
    insertion_code: str
    amino_acid: str
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class Pocket:
    rank: int
    score: float
    center_x: float
    center_y: float
    center_z: float
    residues: Tuple[int, ...]
    mean_probability: float
    max_probability: float


@dataclass(frozen=True)
class PredictionArtifacts:
    job_id: str
    job_dir: Path
    predictions_csv: Path
    pockets_csv: Path
    structure_pdb: Path
    summary_json: Path
    result_zip: Path
    warnings: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelMetadata:
    tasks: Tuple[str, ...]
    thresholds: Dict[str, float]
    gate_weights: Dict[str, Tuple[float, ...]]
    bundle_hash: str
    model_name: str
    source_checkpoint_sha256: Optional[str] = None

