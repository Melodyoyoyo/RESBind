from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch

from .features import SURFACE_COLUMNS, select_device
from . import architecture as model_architecture


PUBLIC_TASKS = {"metal": "metal", "nuclear": "nucleic_acid", "small": "small_molecule"}


class ArrayScaler:
    def __init__(self, path: Path):
        data = np.load(path, allow_pickle=False)
        self.mean = data["mean"].astype(np.float64)
        self.scale = data["scale"].astype(np.float64)
        self.n_features = int(data["n_features"])
        if self.mean.shape != (self.n_features,) or self.scale.shape != (self.n_features,):
            raise ValueError(f"Invalid scaler arrays in {path}")

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values)
        if values.ndim != 2 or values.shape[1] != self.n_features:
            raise ValueError(f"Scaler expected (*, {self.n_features}), got {values.shape}")
        return (values - self.mean) / self.scale


def _file_hash(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _bidirectional_edges(edge_i: np.ndarray, edge_j: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if not len(edge_i):
        return edge_i.astype(int), edge_j.astype(int)
    pairs = np.concatenate([np.column_stack([edge_i, edge_j]), np.column_stack([edge_j, edge_i])])
    pairs = np.unique(pairs[pairs[:, 0] != pairs[:, 1]], axis=0)
    return pairs[:, 0].astype(int), pairs[:, 1].astype(int)


def _pool(values: np.ndarray, edge_i: np.ndarray, edge_j: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pooled = np.zeros_like(values, dtype=np.float32)
    degree = np.zeros(values.shape[0], dtype=np.float32)
    if len(edge_i):
        np.add.at(pooled, edge_i, values[edge_j])
        np.add.at(degree, edge_i, 1.0)
    return pooled / np.maximum(degree[:, None], 1.0), degree


def build_contact_features(prot: np.ndarray, surface: np.ndarray, contact_npz: Path) -> np.ndarray:
    data = np.load(contact_npz, allow_pickle=False)
    residue_numbers = data["residue_number"].astype(int)
    edge_i, edge_j = _bidirectional_edges(data["edge_i"].astype(int), data["edge_j"].astype(int))
    if residue_numbers.tolist() != list(range(1, len(prot) + 1)):
        raise ValueError("Contact-map residues are not exactly 1..L.")
    pooled_prot, degree = _pool(prot, edge_i, edge_j)
    pooled_surface, _ = _pool(surface, edge_i, edge_j)
    available = np.ones((len(prot), 1), dtype=np.float32)
    return np.concatenate([pooled_prot, pooled_surface, degree[:, None], available], axis=1).astype(np.float32)


class V6Predictor:
    def __init__(self, bundle_dir: Path, device: str = "auto"):
        self.bundle_dir = Path(bundle_dir)
        manifest_path = self.bundle_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing V6 inference bundle: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != 1:
            raise RuntimeError("Unsupported V6 inference-bundle schema.")
        if self.manifest.get("bidirectional_contacts") is not True:
            raise RuntimeError("This package requires the bidirectional V6 bundle.")
        files = self.manifest["files"]
        self.raw_surface_scaler = ArrayScaler(self.bundle_dir / files["raw_surface_scaler"])
        self.view_prot_scaler = ArrayScaler(self.bundle_dir / files["view_prot_scaler"])
        self.view_surface_scaler = ArrayScaler(self.bundle_dir / files["view_surface_scaler"])
        self.view_contact_scaler = ArrayScaler(self.bundle_dir / files["view_contact_scaler"])
        module = model_architecture
        dimensions = self.manifest["dimensions"]
        architecture = self.manifest["architecture"]
        self.device = select_device(device)
        self.model = module.MultitaskShallowSharedGatedMLP(
            prot_dim=int(dimensions["prot"]),
            surface_dim=int(dimensions["surface"]),
            contact_dim=int(dimensions["contact"]),
            view_dim=int(architecture["view_dim"]),
            view_hidden_dim=int(architecture["view_hidden_dim"]),
            classifier_hidden_dim=int(architecture["classifier_hidden_dim"]),
            dropout=float(architecture["dropout"]),
            n_tasks=3,
            disabled_views=tuple(architecture.get("disabled_views", [])),
            gate_mode=str(architecture.get("gate_mode", "learned")),
            sharing_mode=str(architecture.get("sharing_mode", "shared")),
        ).to(self.device)
        checkpoint = torch.load(
            self.bundle_dir / self.manifest["files"]["model"], map_location="cpu", weights_only=True
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.bundle_hash = _file_hash(manifest_path)

    @property
    def thresholds(self) -> Dict[str, float]:
        return {PUBLIC_TASKS[key]: float(value) for key, value in self.manifest["thresholds"].items()}

    @property
    def gate_weights(self) -> Dict[str, Tuple[float, ...]]:
        raw = self.model.gate_weights()
        return {PUBLIC_TASKS[key]: tuple(float(v) for v in values) for key, values in raw.items()}

    def predict(
        self,
        protein_id: str,
        sequence: str,
        embedding: np.ndarray,
        surface_frame: pd.DataFrame,
        contact_npz: Path,
    ) -> pd.DataFrame:
        length = len(sequence)
        if embedding.shape != (length, int(self.manifest["dimensions"]["prot"])):
            raise ValueError(f"Unexpected ProtT5 matrix shape: {embedding.shape}")
        columns = list(self.manifest["surface_columns"])
        if columns != SURFACE_COLUMNS:
            raise RuntimeError("Bundle surface-column order differs from the server implementation.")
        surface_raw = surface_frame[columns].to_numpy(dtype=np.float32)
        surface = self.raw_surface_scaler.transform(surface_raw).astype(np.float32)
        contact = build_contact_features(embedding.astype(np.float32), surface, contact_npz)
        prot_scaled = self.view_prot_scaler.transform(embedding).astype(np.float32)
        surface_scaled = self.view_surface_scaler.transform(surface).astype(np.float32)
        contact_scaled = self.view_contact_scaler.transform(contact).astype(np.float32)
        with torch.inference_mode():
            logits = self.model(
                torch.from_numpy(prot_scaled).to(self.device),
                torch.from_numpy(surface_scaled).to(self.device),
                torch.from_numpy(contact_scaled).to(self.device),
            )
            probabilities = torch.sigmoid(logits).cpu().numpy()
        out = pd.DataFrame({"protein_id": protein_id, "position": np.arange(1, length + 1), "amino_acid": list(sequence)})
        for index, internal in enumerate(self.manifest["tasks"]):
            public = PUBLIC_TASKS[internal]
            threshold = float(self.manifest["thresholds"][internal])
            out[f"prob_{public}"] = probabilities[:, index]
            out[f"threshold_{public}"] = threshold
            out[f"pred_{public}"] = probabilities[:, index] >= threshold
        return out


class V6EnsemblePredictor:
    """Compatibility class name: load three models, export independent probabilities."""
    formal_pockets = True

    def __init__(self, bundle_dir: Path, device: str = 'auto'):
        self.bundle_dir = Path(bundle_dir)
        self.manifest = json.loads((self.bundle_dir / 'manifest.json').read_text(encoding='utf-8'))
        if self.manifest.get('schema_version') != 2:
            raise ValueError('Expected three-model bundle schema 2')
        self.members = {seed: V6Predictor(self.bundle_dir / relative, device)
                        for seed, relative in self.manifest['members'].items()}
        if set(self.members) != {'41','42','43'}:
            raise ValueError('Independent evaluation requires seeds 41/42/43')
        self.bundle_hash = hashlib.sha256(json.dumps({
            'multiseed': self.manifest,
            'member_manifests': {s:m.bundle_hash for s,m in self.members.items()},
            'checkpoints': {s:_file_hash(m.bundle_dir / m.manifest['files']['model']) for s,m in self.members.items()},
        }, sort_keys=True).encode()).hexdigest()

    @property
    def thresholds(self):
        return self.members['42'].thresholds

    @property
    def gate_weights(self):
        return self.members['42'].gate_weights

    @property
    def pocket_thresholds(self):
        return {'metal': None, 'nucleic_acid': .767058, 'small_molecule': .8118499}

    def predict(self, protein_id, sequence, embedding, surface_frame, contact_npz):
        results = {s:m.predict(protein_id,sequence,embedding,surface_frame,contact_npz)
                   for s,m in self.members.items()}
        out = results['42'].copy()
        for task in PUBLIC_TASKS.values():
            for seed, frame in results.items():
                pd.testing.assert_frame_equal(out[['protein_id','position','amino_acid']],
                                              frame[['protein_id','position','amino_acid']])
                out[f'prob_{task}_seed{seed}'] = frame[f'prob_{task}'].to_numpy()
            out[f'pocket_prob_{task}'] = out[f'prob_{task}_seed42']
        return out


def load_predictor(bundle_dir: Path, device: str = 'auto'):
    meta = json.loads((Path(bundle_dir) / 'manifest.json').read_text(encoding='utf-8'))
    return (V6EnsemblePredictor if meta.get('schema_version') == 2 else V6Predictor)(bundle_dir, device)
