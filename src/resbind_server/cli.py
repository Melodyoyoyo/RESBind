from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .doctor import run_doctor
from .features import ESMFoldProvider, NetSurfPProvider, ProtT5Provider
from .pipeline import PipelineServices, PredictionPipeline
from .settings import Settings
from .v6 import V6Predictor


def build_pipeline(settings: Settings) -> PredictionPipeline:
    services = PipelineServices(
        prott5=ProtT5Provider(settings.device),
        esmfold=ESMFoldProvider(settings.device, settings.esmfold_chunk_size),
        netsurfp=NetSurfPProvider(settings),
        predictor=V6Predictor(settings.bundle_dir, settings.device),
    )
    return PredictionPipeline(services, settings.job_root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="resbind")
    subcommands = parser.add_subparsers(dest="command", required=True)
    predict = subcommands.add_parser("predict", help="Run one sequence through the complete RESBind pipeline.")
    predict.add_argument("--fasta", type=Path, required=True)
    predict.add_argument("--pdb", type=Path)
    predict.add_argument("--netsurfp-csv", type=Path, help="Import a user-obtained NetSurfP CSV; skip local NetSurfP execution.")
    predict.add_argument("--prott5-npz", type=Path, help="Import sequence-bound ProtT5 features (sequence and embedding arrays); skip ProtT5 execution.")
    predict.add_argument("--out-dir", type=Path)
    predict.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    doctor = subcommands.add_parser("doctor", help="Check models, dependencies, and writable runtime paths.")
    doctor.add_argument("--netsurfp-csv", type=Path, help="Check CSV availability instead of a local NetSurfP command; predict validates its contents.")
    serve = subcommands.add_parser("serve", help="Launch the local Gradio interface.")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    return parser


def main() -> None:
    args = _parser().parse_args()
    settings = Settings.from_env()
    if args.command == "doctor":
        raise SystemExit(0 if run_doctor(settings, netsurfp_csv=args.netsurfp_csv) else 1)
    if args.command == "serve":
        from .app import main as serve_main

        serve_main(host=args.host, port=args.port)
        return
    if args.device:
        settings = Settings(**{**settings.__dict__, "device": args.device})
    try:
        if args.netsurfp_csv is None and not settings.netsurfp_command:
            raise ValueError("Provide --netsurfp-csv from the official service, or configure an authorized local NetSurfP command.")
        pipeline = build_pipeline(settings)
        result = pipeline.run(
            args.fasta.read_text(encoding="utf-8"),
            pdb_path=args.pdb,
            netsurfp_csv=args.netsurfp_csv,
            prott5_npz=args.prott5_npz,
            progress=lambda value: print(value, file=sys.stderr, flush=True),
        )
        if args.out_dir:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            for source in [result.predictions_csv, result.pockets_csv, result.structure_pdb, result.summary_json, result.result_zip]:
                (args.out_dir / source.name).write_bytes(source.read_bytes())
            print(args.out_dir / result.summary_json.name)
        else:
            print(result.summary_json)
    except ValueError as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except Exception as exc:
        print(f"RESBind failed: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

