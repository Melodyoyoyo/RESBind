"""Local stdio tools; no hosted endpoint or implicit feature generation."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import uuid
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from .workflow import inspect_compute, inspect_input, workspace_file
# Import numerical extensions on the main thread before SDK worker threads run.
from . import features, structure
import torch

mcp = FastMCP("RESBind")
_lock = threading.Lock()


def _workspace() -> Path:
    value = os.environ.get("RESBIND_MCP_WORKSPACE")
    if not value:
        raise ValueError("Set RESBIND_MCP_WORKSPACE to an existing dedicated directory.")
    root = Path(value).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Workspace must be a directory.")
    return root


def _inputs(fasta, pdb=None, netsurfp_csv=None, prott5_npz=None):
    root = _workspace()
    paths = {}
    for name, value, limit in [("fasta", fasta, 65536), ("pdb", pdb, 10000000),
                               ("netsurfp", netsurfp_csv, 10000000),
                               ("prott5", prott5_npz, 20000000)]:
        paths[name] = workspace_file(root, value, limit) if value is not None else None
    return paths


@mcp.tool()
def check_environment() -> dict:
    """Inspect this machine without installing software or loading feature models."""
    return inspect_compute()


@mcp.tool()
def inspect_fasta(fasta: str, mode: str = "predict", pdb: str | None = None,
                  netsurfp_csv: str | None = None, prott5_npz: str | None = None) -> dict:
    """Check one FASTA and optional sequence-matched features inside the workspace.

    Evaluation/training are inspection-only; this tool does not split data.
    """
    paths = _inputs(fasta, pdb, netsurfp_csv, prott5_npz)
    return inspect_input(paths.pop("fasta").read_text(encoding="utf-8"), mode, **paths)


@mcp.tool()
def predict_saved(fasta: str, pdb: str, netsurfp_csv: str, prott5_npz: str) -> dict:
    """Run V6 on CPU with ALL features already supplied. Writes a new result directory.

    PDB must be an ESMFold prediction with matching sequence; provenance is the
    caller's responsibility. No downloads, training or remote jobs. Synchronous,
    one job at a time, 180-second timeout. Pocket output is small-molecule-only.
    """
    if not _lock.acquire(blocking=False):
        raise ValueError("A prediction is running. Retry after it finishes.")
    try:
        paths = _inputs(fasta, pdb, netsurfp_csv, prott5_npz)
        inspection = inspect_input(paths["fasta"].read_text(encoding="utf-8"),
                                   **{k: v for k, v in paths.items() if k != "fasta"})
        if not inspection["ready_for_saved_prediction"]:
            raise ValueError("All three feature inputs are required.")
        bundle = os.environ.get("RESBIND_BUNDLE_DIR")
        if not bundle or not (Path(bundle) / "manifest.json").is_file():
            raise ValueError("Owner must configure RESBIND_BUNDLE_DIR.")
        root = _workspace()
        # New private run directory avoids trusting or overwriting shared caches.
        run = root / ("resbind_run_" + uuid.uuid4().hex)
        run.mkdir(exist_ok=False)
        env = os.environ.copy()
        env.update(RESBIND_JOB_ROOT=str(run / "jobs"), RESBIND_CACHE_ROOT=str(run / "cache"),
                   RESBIND_BUNDLE_DIR=str(Path(bundle).resolve()), HF_HUB_OFFLINE="1",
                   TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        env.pop("RESBIND_NETSURFP_COMMAND_JSON", None)
        command = [sys.executable, "-I", "-m", "resbind_server.cli", "predict",
                   "--fasta", str(paths["fasta"]), "--pdb", str(paths["pdb"]),
                   "--netsurfp-csv", str(paths["netsurfp"]), "--prott5-npz", str(paths["prott5"]),
                   "--device", "cpu", "--out-dir", str(run / "results")]
        result = subprocess.run(command, env=env, cwd=root, capture_output=True, stdin=subprocess.DEVNULL,
                                text=True, timeout=180, check=False)
        if result.returncode:
            raise ValueError("Prediction failed: " + result.stderr[-2000:])
        summary = json.loads((run / "results" / "summary.json").read_text(encoding="utf-8"))
        return {"summary": summary, "artifacts": {p.name: str(p) for p in (run / "results").iterdir()},
                "scope": "Three residue tasks; legacy small-molecule pockets only. Predictions are not experimental evidence."}
    finally:
        _lock.release()


def main():
    _workspace()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
