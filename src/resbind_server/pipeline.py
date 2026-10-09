from __future__ import annotations

import hashlib
import json
import os
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .domain import PredictionArtifacts
from .fasta import parse_single_fasta
from .features import load_netsurfp_table, load_prott5_npz
from .structure import (
    cluster_small_molecule_pockets,
    coordinates,
    parse_pdb_ca,
    validate_structure_sequence,
    write_contact_npz,
)


Progress = Callable[[str], None]


@dataclass
class PipelineServices:
    prott5: object
    esmfold: object
    netsurfp: object
    predictor: object


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class PredictionPipeline:
    stages = (
        "Queued", "Validating input", "Generating structure", "Generating ProtT5 features",
        "Generating surface features", "Building contacts", "Predicting binding residues",
        "Clustering pockets", "Packaging results", "Complete",
    )

    def __init__(self, services: PipelineServices, job_root: Path):
        self.services = services
        self.job_root = Path(job_root)

    def run(
        self,
        fasta_text: str,
        pdb_path: Optional[Path] = None,
        progress: Optional[Progress] = None,
        *,
        netsurfp_csv: Optional[Path] = None,
        prott5_npz: Optional[Path] = None,
    ) -> PredictionArtifacts:
        report = progress or (lambda _stage: None)
        timings = {}

        def stage(name: str):
            report(name)
            timings[name] = time.perf_counter()

        stage("Queued")
        stage("Validating input")
        record = parse_single_fasta(fasta_text)
        pdb_bytes = Path(pdb_path).read_bytes() if pdb_path else b""
        csv_bytes = Path(netsurfp_csv).read_bytes() if netsurfp_csv is not None else None
        npz_bytes = Path(prott5_npz).read_bytes() if prott5_npz is not None else None
        # Validate imports before any heavyweight providers or completed-cache reuse.
        imported_surface = load_netsurfp_table(netsurfp_csv, record.protein_id, record.sequence, content=csv_bytes) if csv_bytes is not None else None
        imported_embedding = load_prott5_npz(prott5_npz, record.sequence, content=npz_bytes) if npz_bytes is not None else None
        bundle_hash = str(getattr(self.services.predictor, "bundle_hash", "unbundled"))
        source_inputs = {
            "netsurfp": {"mode": "imported_csv" if csv_bytes is not None else "configured_provider", "sha256": _sha256_bytes(csv_bytes) if csv_bytes is not None else None},
            "prott5": {"mode": "imported_npz" if npz_bytes is not None else "generated", "sha256": _sha256_bytes(npz_bytes) if npz_bytes is not None else None},
            "structure": {"mode": "uploaded" if pdb_path else "generated", "sha256": _sha256_bytes(pdb_bytes) if pdb_path else None},
        }
        job_key = json.dumps({
            "pipeline_schema": 2, "protein_id": record.protein_id,
            "sequence": record.sequence, "bundle_hash": bundle_hash, "inputs": source_inputs,
        }, sort_keys=True).encode()
        job_id = _sha256_bytes(job_key)[:16]
        job_dir = self.job_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        summary_path = job_dir / "summary.json"
        complete_path = job_dir / "resbind_results.zip"
        if summary_path.exists() and complete_path.exists():
            report("Complete")
            return self._artifacts(job_id, job_dir)
        _atomic_text(job_dir / "input.fasta", f">{record.protein_id}\n{record.sequence}\n")

        stage("Generating structure")
        structure_path = job_dir / "structure.pdb"
        structure_source = "uploaded"
        if pdb_path:
            structure_path.write_bytes(pdb_bytes)
        elif not structure_path.exists():
            structure_source = "ESMFold"
            _atomic_text(structure_path, self.services.esmfold.fold(record.sequence))
        else:
            structure_source = "ESMFold cache"
        residues = parse_pdb_ca(structure_path)
        validate_structure_sequence(residues, record.sequence)

        stage("Generating ProtT5 features")
        embedding_path = job_dir / "prott5.npy"
        if imported_embedding is not None:
            embedding = imported_embedding
        elif embedding_path.exists():
            embedding = np.load(embedding_path, allow_pickle=False).astype(np.float32)
        else:
            embedding = self.services.prott5.embed(record.sequence)
            temporary = embedding_path.with_suffix(".npy.tmp")
            with temporary.open("wb") as handle:
                np.save(handle, embedding)
            os.replace(temporary, embedding_path)

        stage("Generating surface features")
        surface_csv = job_dir / "surface_features.csv"
        if surface_csv.exists():
            surface = pd.read_csv(surface_csv)
        else:
            residue_table = imported_surface if imported_surface is not None else self.services.netsurfp.predict(record.protein_id, record.sequence, job_dir)
            from .features import summarize_surface

            surface = summarize_surface(record.protein_id, record.sequence, residue_table)
            _atomic_csv(surface_csv, surface)

        stage("Building contacts")
        contacts_path = job_dir / "contacts_10a.npz"
        if not contacts_path.exists():
            write_contact_npz(contacts_path, residues, cutoff=10.0)

        stage("Predicting binding residues")
        predictions_path = job_dir / "residue_predictions.csv"
        predictions = self.services.predictor.predict(
            record.protein_id, record.sequence, embedding, surface, contacts_path
        )
        _atomic_csv(predictions_path, predictions)

        stage("Clustering pockets")
        small_threshold = float(self.services.predictor.thresholds["small_molecule"])
        pockets = cluster_small_molecule_pockets(
            coordinates(residues), predictions["prob_small_molecule"].to_numpy(), small_threshold, cutoff=6.0
        )
        pocket_frame = pd.DataFrame(
            [
                {
                    "rank": p.rank, "score": p.score, "size": len(p.residues),
                    "center_x": p.center_x, "center_y": p.center_y, "center_z": p.center_z,
                    "mean_probability": p.mean_probability, "max_probability": p.max_probability,
                    "residues": ";".join(map(str, p.residues)),
                }
                for p in pockets
            ],
            columns=["rank", "score", "size", "center_x", "center_y", "center_z", "mean_probability", "max_probability", "residues"],
        )
        pockets_path = job_dir / "pockets.csv"
        _atomic_csv(pockets_path, pocket_frame)

        stage("Packaging results")
        end = time.perf_counter()
        start_times = list(timings.items())
        runtime = {}
        for index, (name, started) in enumerate(start_times):
            stopped = start_times[index + 1][1] if index + 1 < len(start_times) else end
            runtime[name] = round(stopped - started, 3)
        summary = {
            "job_id": job_id,
            "protein_id": record.protein_id,
            "sequence_length": len(record.sequence),
            "sequence_sha256": _sha256_bytes(record.sequence.encode()),
            "structure_source": structure_source,
            "pipeline_schema": 2,
            "input_sources": source_inputs,
            "model_name": getattr(self.services.predictor, "manifest", {}).get("model_name", "RESBind V6"),
            "bundle_hash": bundle_hash,
            "thresholds": self.services.predictor.thresholds,
            "gate_weights": self.services.predictor.gate_weights,
            "positive_residue_counts": {
                task: int(predictions[f"pred_{task}"].sum())
                for task in ("metal", "nucleic_acid", "small_molecule")
            },
            "n_small_molecule_pockets": len(pockets),
            "runtime_seconds": runtime,
            "warnings": ["Computational predictions require experimental validation."],
        }
        _atomic_text(summary_path, json.dumps(summary, indent=2))
        archive_tmp = complete_path.with_suffix(".zip.tmp")
        with zipfile.ZipFile(archive_tmp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in ["input.fasta", "structure.pdb", "residue_predictions.csv", "pockets.csv", "summary.json"]:
                archive.write(job_dir / name, arcname=name)
        os.replace(archive_tmp, complete_path)
        report("Complete")
        return self._artifacts(job_id, job_dir)

    @staticmethod
    def _artifacts(job_id: str, job_dir: Path) -> PredictionArtifacts:
        return PredictionArtifacts(
            job_id=job_id,
            job_dir=job_dir,
            predictions_csv=job_dir / "residue_predictions.csv",
            pockets_csv=job_dir / "pockets.csv",
            structure_pdb=job_dir / "structure.pdb",
            summary_json=job_dir / "summary.json",
            result_zip=job_dir / "resbind_results.zip",
        )

