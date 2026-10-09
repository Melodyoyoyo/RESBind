from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .domain import Pocket, StructureResidue


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "X", "PYL": "X", "ASX": "X", "GLX": "X",
    "UNK": "X",
}


class StructureMismatchError(ValueError):
    """Raised when a structure cannot be mapped one-to-one to the query."""


def _normalise_aa(letter: str) -> str:
    return "X" if letter in "BZUOJ" else letter


def parse_pdb_ca(pdb_path: Path) -> Tuple[StructureResidue, ...]:
    by_chain: Dict[str, List[Tuple[int, str, str, float, float, float]]] = {}
    seen: set = set()
    with Path(pdb_path).open("rt", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM  ") or line[12:16].strip() != "CA":
                continue
            if line[16].strip() not in {"", "A"}:
                continue
            chain = line[21].strip() or "_"
            icode = line[26].strip()
            try:
                resnum = int(line[22:26])
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except ValueError:
                continue
            key = (chain, resnum, icode)
            if key in seen:
                continue
            seen.add(key)
            amino_acid = AA3_TO_1.get(line[17:20].strip().upper())
            if amino_acid is None:
                continue
            by_chain.setdefault(chain, []).append((resnum, icode, amino_acid, x, y, z))
    if not by_chain:
        raise StructureMismatchError("No protein C-alpha atoms were found in the PDB file.")
    first_chain = next(iter(by_chain))
    values = by_chain[first_chain]
    return tuple(
        StructureResidue(index, first_chain, resnum, icode, aa, x, y, z)
        for index, (resnum, icode, aa, x, y, z) in enumerate(values, 1)
    )


def validate_structure_sequence(residues: Sequence[StructureResidue], sequence: str) -> None:
    if len(residues) != len(sequence):
        raise StructureMismatchError(
            f"Structure chain has {len(residues)} C-alpha residues but the sequence has {len(sequence)} residues."
        )
    mismatches = []
    for residue, expected in zip(residues, sequence):
        observed = _normalise_aa(residue.amino_acid)
        wanted = _normalise_aa(expected)
        if observed != wanted and observed != "X" and wanted != "X":
            mismatches.append(f"{residue.sequence_position}:{observed}!={wanted}")
    if mismatches:
        preview = ", ".join(mismatches[:5])
        suffix = " ..." if len(mismatches) > 5 else ""
        raise StructureMismatchError(f"Structure sequence does not match the submitted sequence ({preview}{suffix}).")


def coordinates(residues: Sequence[StructureResidue]) -> np.ndarray:
    return np.asarray([(r.x, r.y, r.z) for r in residues], dtype=np.float32)


def contact_edges(coords: np.ndarray, cutoff: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("Coordinates must have shape (L, 3).")
    delta = coords[:, None, :] - coords[None, :, :]
    distance_sq = np.einsum("ijk,ijk->ij", delta, delta)
    adjacent = distance_sq <= float(cutoff) ** 2
    np.fill_diagonal(adjacent, False)
    edge_i, edge_j = np.where(np.triu(adjacent, k=1))
    degree = adjacent.sum(axis=1).astype(np.int32)
    return edge_i.astype(np.int32), edge_j.astype(np.int32), degree


def write_contact_npz(path: Path, residues: Sequence[StructureResidue], cutoff: float = 10.0) -> Path:
    coords = coordinates(residues)
    edge_i, edge_j, degree = contact_edges(coords, cutoff)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        cutoff_angstrom=np.asarray(cutoff, dtype=np.float32),
        residue_number=np.arange(1, len(residues) + 1, dtype=np.int32),
        pdb_residue_number=np.asarray([r.residue_number for r in residues], dtype=np.int32),
        insertion_code=np.asarray([r.insertion_code for r in residues]),
        chain=np.asarray([r.chain for r in residues]),
        ca_coords=coords,
        edge_i=edge_i,
        edge_j=edge_j,
        degree=degree,
    )
    return path


def cluster_small_molecule_pockets(
    coords: np.ndarray,
    probabilities: Sequence[float],
    threshold: float,
    cutoff: float = 6.0,
) -> Tuple[Pocket, ...]:
    coords = np.asarray(coords, dtype=np.float32)
    probs = np.asarray(probabilities, dtype=np.float64)
    if len(coords) != len(probs):
        raise ValueError("Coordinate and probability lengths differ.")
    positive = np.flatnonzero(probs >= threshold)
    if not len(positive):
        return ()
    selected = coords[positive]
    delta = selected[:, None, :] - selected[None, :, :]
    adjacent = np.einsum("ijk,ijk->ij", delta, delta) <= float(cutoff) ** 2
    np.fill_diagonal(adjacent, False)
    unvisited = set(range(len(positive)))
    components: List[List[int]] = []
    while unvisited:
        seed = min(unvisited)
        unvisited.remove(seed)
        stack = [seed]
        component = [seed]
        while stack:
            current = stack.pop()
            neighbors = [int(v) for v in np.flatnonzero(adjacent[current]) if int(v) in unvisited]
            for neighbor in neighbors:
                unvisited.remove(neighbor)
                stack.append(neighbor)
                component.append(neighbor)
        components.append(sorted(component))

    raw = []
    for component in components:
        sequence_indices = positive[np.asarray(component, dtype=int)]
        values = probs[sequence_indices]
        weights = values / values.sum() if values.sum() > 0 else np.full(len(values), 1 / len(values))
        center = np.average(coords[sequence_indices], axis=0, weights=weights)
        raw.append(
            {
                "score": float(np.maximum(values - threshold, 0.0).sum()),
                "center": center,
                "residues": tuple(int(i) + 1 for i in sequence_indices),
                "mean": float(values.mean()),
                "max": float(values.max()),
            }
        )
    raw.sort(key=lambda item: (-item["score"], -item["max"], item["residues"][0]))
    return tuple(
        Pocket(
            rank=rank,
            score=item["score"],
            center_x=float(item["center"][0]),
            center_y=float(item["center"][1]),
            center_z=float(item["center"][2]),
            residues=item["residues"],
            mean_probability=item["mean"],
            max_probability=item["max"],
        )
        for rank, item in enumerate(raw, 1)
    )

