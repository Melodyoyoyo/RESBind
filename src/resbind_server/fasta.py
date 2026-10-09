from __future__ import annotations

import re
from typing import List, Tuple

from .domain import ProteinInput


_ALLOWED = frozenset("ACDEFGHIKLMNPQRSTVWYX")
_AMBIGUOUS = str.maketrans({letter: "X" for letter in "BZUOJ"})
_NUCLEOTIDE = frozenset("ACGTUN")


def _safe_identifier(header: str, default_id: str) -> str:
    first = header.strip().split()[0] if header.strip() else default_id
    parts = first.split("|")
    if len(parts) >= 2 and parts[1]:
        first = parts[1]
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", first).strip("._-")
    return value[:80] or default_id


def _split_records(text: str) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    header = None
    chunks: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(chunks)))
            header = line[1:].strip()
            chunks = []
        else:
            if header is None and records:
                raise ValueError("Sequence data appeared outside a FASTA record.")
            chunks.append(re.sub(r"\s+", "", line))
    if header is not None:
        records.append((header, "".join(chunks)))
    elif chunks:
        records.append(("", "".join(chunks)))
    return records


def parse_single_fasta(text: str, default_id: str = "query") -> ProteinInput:
    records = _split_records(text)
    if not records or not records[0][1]:
        raise ValueError("Enter one protein sequence or one FASTA record.")
    if len(records) != 1:
        raise ValueError("The prototype accepts exactly one FASTA record per job.")
    header, raw_sequence = records[0]
    sequence = raw_sequence.upper().translate(_AMBIGUOUS)
    invalid = sorted(set(sequence) - _ALLOWED)
    if invalid:
        raise ValueError("Invalid protein-sequence characters: " + ", ".join(invalid))
    if len(sequence) < 20:
        raise ValueError("Protein sequence must contain at least 20 residues.")
    if len(sequence) > 1024:
        raise ValueError("Protein sequence exceeds the prototype limit of 1,024 residues.")
    if len(sequence) >= 20 and set(sequence) <= _NUCLEOTIDE:
        raise ValueError("Input appears to be a nucleotide sequence, not a protein sequence.")
    return ProteinInput(_safe_identifier(header, default_id), sequence)

