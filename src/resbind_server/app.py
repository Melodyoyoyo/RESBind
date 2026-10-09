from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from .cli import build_pipeline
from .render import prediction_figure, structure_html
from .settings import Settings


_PIPELINE = None


def _path_from_upload(value) -> Optional[Path]:
    if value is None:
        return None
    if isinstance(value, str):
        return Path(value)
    path = getattr(value, "name", None) or getattr(value, "path", None)
    return Path(path) if path else None


def _get_pipeline(settings: Settings):
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = build_pipeline(settings)
    return _PIPELINE


def create_app(settings: Optional[Settings] = None):
    import gradio as gr

    settings = settings or Settings.from_env()

    def submit(sequence_text, fasta_upload, pdb_upload, progress=gr.Progress()):
        fasta_path = _path_from_upload(fasta_upload)
        pdb_path = _path_from_upload(pdb_upload)
        text = fasta_path.read_text(encoding="utf-8") if fasta_path else sequence_text
        if not text or not text.strip():
            raise gr.Error("Enter a protein sequence or upload one FASTA file.")
        pipeline = _get_pipeline(settings)
        stage_to_fraction = {name: index / (len(pipeline.stages) - 1) for index, name in enumerate(pipeline.stages)}

        def report(stage):
            progress(stage_to_fraction.get(stage, 0), desc=stage)

        try:
            artifacts = pipeline.run(text, pdb_path=pdb_path, progress=report)
        except Exception as exc:
            raise gr.Error(str(exc)) from exc
        predictions = pd.read_csv(artifacts.predictions_csv)
        pockets = pd.read_csv(artifacts.pockets_csv)
        summary = json.loads(artifacts.summary_json.read_text(encoding="utf-8"))
        positive = predictions[
            predictions[["pred_metal", "pred_nucleic_acid", "pred_small_molecule"]].any(axis=1)
        ].copy()
        downloads = [
            str(artifacts.predictions_csv), str(artifacts.pockets_csv), str(artifacts.structure_pdb),
            str(artifacts.summary_json), str(artifacts.result_zip),
        ]
        return (
            "Complete",
            json.dumps(summary, indent=2),
            prediction_figure(predictions),
            structure_html(artifacts.structure_pdb, predictions),
            positive,
            pockets,
            downloads,
        )

    with gr.Blocks(title="RESBind") as demo:
        gr.Markdown(
            "# RESBind\n"
            "Predict metal-, nucleic-acid-, and small-molecule-binding residues from one protein sequence. "
            "Hits are grouped into task-wise candidate pockets (6/8/6 Å); isolated residues are labeled separately."
        )
        with gr.Row():
            with gr.Column(scale=2):
                sequence = gr.Textbox(label="Protein sequence or FASTA", lines=9, placeholder=">query\nM...")
                fasta = gr.File(label="Optional FASTA upload", file_types=[".fasta", ".fa", ".faa", ".txt"])
                pdb = gr.File(label="Optional structure (PDB)", file_types=[".pdb"])
                run = gr.Button("Run RESBind", variant="primary")
                status = gr.Textbox(label="Status", interactive=False)
            with gr.Column(scale=1):
                summary = gr.Code(label="Job summary", language="json")
        tracks = gr.Plot(label="Per-residue predictions")
        viewer = gr.HTML(label="3D structure")
        with gr.Tab("Predicted residues"):
            residues = gr.Dataframe(interactive=False)
        with gr.Tab("Candidate pockets and isolated residues"):
            pockets = gr.Dataframe(interactive=False)
        downloads = gr.File(label="Download results", file_count="multiple")
        run.click(submit, [sequence, fasta, pdb], [status, summary, tracks, viewer, residues, pockets, downloads])
        gr.Markdown(
            "Predictions are computational hypotheses, not experimental evidence. Thresholds were fixed on the archived validation set."
        )
    return demo


def main(host: Optional[str] = None, port: Optional[int] = None) -> None:
    settings = Settings.from_env()
    # Gradio probes its own local URL at launch; bypass corporate/system proxies.
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
    demo = create_app(settings)
    demo.queue(default_concurrency_limit=1, max_size=20).launch(
        server_name=host or settings.host, server_port=port or settings.port, share=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    values = parser.parse_args()
    main(values.host, values.port)
