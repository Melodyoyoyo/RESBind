from __future__ import annotations

import csv
import io
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .settings import Settings


SURFACE_COLUMNS = [
    "protein_length", "distance_to_n_term", "distance_to_c_term", "relative_position",
    "is_n_terminal_window", "is_c_terminal_window", "center_rsa", "center_exposed",
    "center_asa", "center_disorder", "center_p_helix", "center_p_strand", "center_p_coil",
    "window_mean_rsa", "window_max_rsa", "window_min_rsa", "window_std_rsa",
    "window_exposed_ratio", "window_mean_asa", "window_max_asa", "window_mean_disorder",
    "window_max_disorder", "window_mean_p_helix", "window_mean_p_strand", "window_mean_p_coil",
]


def select_device(preference: str):
    import torch

    if preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if preference == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ProtT5Provider:
    model_name = "Rostlab/prot_t5_xl_uniref50"

    def __init__(self, device: str = "auto"):
        self.device_preference = device
        self._model = None
        self._tokenizer = None

    def _load(self):
        import torch
        from transformers import T5EncoderModel, T5Tokenizer

        if self._model is None:
            self.device = select_device(self.device_preference)
            self._tokenizer = T5Tokenizer.from_pretrained(self.model_name, do_lower_case=False)
            self._model = T5EncoderModel.from_pretrained(self.model_name)
            if self.device.type == "cuda":
                self._model = self._model.half()
            self._model = self._model.to(self.device).eval()
        return torch

    def embed(self, sequence: str) -> np.ndarray:
        torch = self._load()
        token_sequence = " ".join(re.sub(r"[UZOB]", "X", sequence))
        encoded = self._tokenizer([token_sequence], add_special_tokens=True, padding=True, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=self.device.type == "cuda"
        ):
            hidden = self._model(**encoded).last_hidden_state[0, : len(sequence)]
        result = hidden.detach().float().cpu().numpy().astype(np.float32)
        if result.shape != (len(sequence), 1024) or not np.isfinite(result).all():
            raise RuntimeError(f"Unexpected ProtT5 embedding shape or values: {result.shape}")
        return result


class ESMFoldProvider:
    model_name = "facebook/esmfold_v1"

    def __init__(self, device: str = "auto", chunk_size: int = 64):
        self.device_preference = device
        self.chunk_size = chunk_size
        self._model = None

    def _load(self):
        import torch
        from transformers import EsmForProteinFolding

        if self._model is None:
            self.device = select_device(self.device_preference)
            self._model = EsmForProteinFolding.from_pretrained(self.model_name, low_cpu_mem_usage=True).eval()
            if hasattr(self._model, "trunk"):
                self._model.trunk.set_chunk_size(self.chunk_size)
            if self.device.type == "cuda":
                self._model = self._model.cuda()
                if hasattr(self._model, "esm"):
                    self._model.esm = self._model.esm.half()
            else:
                self._model = self._model.cpu()
        return torch

    def fold(self, sequence: str) -> str:
        torch = self._load()
        with torch.inference_mode():
            pdb_text = self._model.infer_pdb(sequence)
        if not pdb_text.strip().startswith("PARENT") and "ATOM" not in pdb_text:
            raise RuntimeError("ESMFold returned no PDB atoms.")
        return pdb_text


def _extract_identifier(value: str) -> str:
    tokens = str(value).strip().lstrip(">").split()
    if not tokens:
        raise ValueError("NetSurfP output contains an empty protein identifier.")
    first = tokens[0]
    parts = first.split("|")
    return parts[1] if len(parts) >= 2 and parts[1] else first


def robust_read_netsurfp_csv(path: Path, *, content: Optional[bytes] = None) -> pd.DataFrame:
    rows: List[List[str]] = []
    data = Path(path).read_bytes() if content is None else content
    with io.StringIO(data.decode("utf-8-sig"), newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = [value.strip() for value in next(reader)]
        except StopIteration as exc:
            raise ValueError("NetSurfP CSV is empty.") from exc
        if len(set(header)) != len(header):
            raise ValueError("Duplicate NetSurfP CSV column names.")
        for row in reader:
            if not row:
                continue
            if len(row) != len(header):
                raise ValueError(f"Malformed NetSurfP row with {len(row)} columns; expected {len(header)}.")
            rows.append(row)
    return pd.DataFrame(rows, columns=header)


def load_netsurfp_table(path: Path, protein_id: str, sequence: str, *, content: Optional[bytes] = None) -> pd.DataFrame:
    raw = robust_read_netsurfp_csv(path, content=content)
    raw.columns = [str(value).strip() for value in raw.columns]
    if "protein_id" not in raw.columns:
        if "id" not in raw.columns:
            raise ValueError("NetSurfP CSV requires an id or protein_id column.")
        raw["protein_id"] = raw["id"].map(_extract_identifier)
    raw = raw.rename(
        columns={"seq": "aa", "n": "position", "p[q3_H]": "p_helix", "p[q3_E]": "p_strand", "p[q3_C]": "p_coil"}
    )
    if raw.columns.duplicated().any():
        raise ValueError("Duplicate NetSurfP columns after normalizing aliases.")
    required = ["protein_id", "aa", "position", "rsa", "asa", "disorder", "p_helix", "p_strand", "p_coil"]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"NetSurfP CSV missing columns: {missing}")
    table = raw.loc[raw["protein_id"].astype(str).str.strip().eq(protein_id), required].copy()
    if table.empty and raw["protein_id"].nunique() == 1:
        table = raw[required].copy()
        table["protein_id"] = protein_id
    if table.empty:
        raise ValueError(f"No unambiguous NetSurfP records for protein {protein_id}.")
    for column in ["position", "rsa", "asa", "disorder", "p_helix", "p_strand", "p_coil"]:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    if table.isna().any().any():
        raise ValueError("NetSurfP output contains missing or non-numeric required values.")
    numeric = table[["position", "rsa", "asa", "disorder", "p_helix", "p_strand", "p_coil"]]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("NetSurfP output contains non-finite values.")
    if not table["position"].mod(1).eq(0).all():
        raise ValueError("NetSurfP residue positions must be integers.")
    if table["position"].duplicated().any():
        raise ValueError("Duplicate NetSurfP residue positions; supply one record per residue.")
    probability_columns = ["rsa", "disorder", "p_helix", "p_strand", "p_coil"]
    if not ((table[probability_columns] >= 0) & (table[probability_columns] <= 1)).all().all():
        raise ValueError("NetSurfP RSA and probabilities must be in [0, 1].")
    if (table["asa"] < 0).any():
        raise ValueError("NetSurfP ASA must be non-negative.")
    if not np.allclose(table[["p_helix", "p_strand", "p_coil"]].sum(axis=1), 1, atol=.01, rtol=0):
        raise ValueError("NetSurfP Q3 probabilities must sum to 1 (tolerance 0.01).")
    table["position"] = table["position"].astype(int)
    table = table.sort_values("position")
    if table["position"].tolist() != list(range(1, len(sequence) + 1)):
        raise ValueError(f"NetSurfP positions are not exactly 1..{len(sequence)}.")
    observed = "".join(table["aa"].astype(str).str.strip().str.upper())
    if observed.translate(str.maketrans({letter: "X" for letter in "BZUOJ"})) != sequence:
        raise ValueError("NetSurfP amino-acid sequence does not match the submitted FASTA.")
    return table.set_index("position")


def load_prott5_npz(path: Path, sequence: str, *, content: Optional[bytes] = None) -> np.ndarray:
    """Read sequence-bound cached features without executing pickle content."""
    source = Path(path) if content is None else io.BytesIO(content)
    with np.load(source, allow_pickle=False) as data:
        if "sequence" not in data or "embedding" not in data:
            raise ValueError("ProtT5 NPZ requires sequence and embedding arrays.")
        stored_sequence = data["sequence"]
        if stored_sequence.shape != () or stored_sequence.dtype.kind != "U" or stored_sequence.item() != sequence:
            raise ValueError("ProtT5 NPZ sequence must be a scalar Unicode string matching FASTA exactly.")
        embedding = data["embedding"]
        if embedding.shape != (len(sequence), 1024) or embedding.dtype.kind != "f":
            raise ValueError("ProtT5 embedding must be a floating-point array of shape (sequence length, 1024).")
        if not np.isfinite(embedding).all():
            raise ValueError("ProtT5 embedding contains non-finite values.")
        with np.errstate(over="ignore"):
            embedding = embedding.astype(np.float32)
        if not np.isfinite(embedding).all():
            raise ValueError("ProtT5 embedding values exceed float32 range.")
        return embedding


class NetSurfPProvider:
    def __init__(self, settings: Settings):
        self.settings = settings

    def predict(self, protein_id: str, sequence: str, work_dir: Path) -> pd.DataFrame:
        command = self.settings.netsurfp_command
        if not command:
            raise RuntimeError(
                "NetSurfP is not configured. Set RESBIND_NETSURFP_COMMAND_JSON to an argument array containing "
                "{input_fasta} and {output_csv}."
            )
        executable = shutil.which(command[0]) or (command[0] if Path(command[0]).exists() else None)
        if not executable:
            raise RuntimeError(f"NetSurfP executable was not found: {command[0]}")
        work_dir.mkdir(parents=True, exist_ok=True)
        fasta_path = work_dir / "netsurfp_input.fasta"
        output_path = work_dir / "netsurfp.csv"
        fasta_path.write_text(f">{protein_id}\n{sequence}\n", encoding="utf-8")
        replacements = {"{input_fasta}": str(fasta_path), "{output_csv}": str(output_path)}
        argv = [replacements.get(part, part) for part in command]
        if str(fasta_path) not in argv or str(output_path) not in argv:
            raise RuntimeError("NetSurfP command must include {input_fasta} and {output_csv} as separate arguments.")
        subprocess.run(argv, check=True, timeout=self.settings.netsurfp_timeout_seconds, cwd=str(work_dir))
        if not output_path.exists():
            raise RuntimeError("NetSurfP completed without creating the configured output CSV.")
        return load_netsurfp_table(output_path, protein_id, sequence)


def summarize_surface(protein_id: str, sequence: str, residue: pd.DataFrame) -> pd.DataFrame:
    rows = []
    length = len(sequence)
    for position, amino_acid in enumerate(sequence, 1):
        window = residue.loc[max(1, position - 5) : min(length, position + 5)]
        center = residue.loc[position]
        rows.append(
            {
                "peptide_id": protein_id,
                "center_pos": position,
                "amino_acid": amino_acid,
                "protein_length": length,
                "distance_to_n_term": position - 1,
                "distance_to_c_term": length - position,
                "relative_position": position / length,
                "is_n_terminal_window": int(position <= 13),
                "is_c_terminal_window": int(length - position <= 12),
                "center_rsa": float(center["rsa"]),
                "center_exposed": int(float(center["rsa"]) >= 0.25),
                "center_asa": float(center["asa"]),
                "center_disorder": float(center["disorder"]),
                "center_p_helix": float(center["p_helix"]),
                "center_p_strand": float(center["p_strand"]),
                "center_p_coil": float(center["p_coil"]),
                "window_mean_rsa": float(window["rsa"].mean()),
                "window_max_rsa": float(window["rsa"].max()),
                "window_min_rsa": float(window["rsa"].min()),
                "window_std_rsa": float(window["rsa"].std(ddof=1)) if len(window) > 1 else 0.0,
                "window_exposed_ratio": float((window["rsa"] >= 0.25).mean()),
                "window_mean_asa": float(window["asa"].mean()),
                "window_max_asa": float(window["asa"].max()),
                "window_mean_disorder": float(window["disorder"].mean()),
                "window_max_disorder": float(window["disorder"].max()),
                "window_mean_p_helix": float(window["p_helix"].mean()),
                "window_mean_p_strand": float(window["p_strand"].mean()),
                "window_mean_p_coil": float(window["p_coil"].mean()),
            }
        )
    frame = pd.DataFrame(rows)
    matrix = frame[SURFACE_COLUMNS].to_numpy(dtype=np.float32)
    if matrix.shape != (length, 25) or not np.isfinite(matrix).all():
        raise RuntimeError(f"Invalid V6 surface feature matrix: {matrix.shape}")
    return frame

