"""Command-line entry point for the research prototype."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from claimguard.config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(prog="claimguard")
    commands = parser.add_subparsers(dest="command", required=True)
    api_parser = commands.add_parser("api", help="Run the FastAPI service.")
    api_parser.add_argument("--host", default="127.0.0.1")
    api_parser.add_argument("--port", type=int, default=8000)
    api_parser.add_argument("--reload", action="store_true")
    dashboard_parser = commands.add_parser("dashboard", help="Run the Streamlit dashboard.")
    dashboard_parser.add_argument("--port", type=int, default=8501)
    validate_parser = commands.add_parser(
        "validate-manifest", help="Validate dataset CSV split hygiene."
    )
    validate_parser.add_argument("manifest", type=Path)
    collect_parser = commands.add_parser("collect", help="Collect per-image agent features.")
    collect_parser.add_argument("--manifest", type=Path, required=True)
    collect_parser.add_argument("--output", type=Path, required=True)
    collect_parser.add_argument("--model", default=get_settings().model_path)
    collect_parser.add_argument(
        "--fusion-model", type=Path, default=get_settings().fusion_model_path
    )
    train_parser = commands.add_parser(
        "train-detector", help="Fine-tune a three-class image classifier."
    )
    train_parser.add_argument("--manifest", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--base-model", default="microsoft/resnet-50")
    train_parser.add_argument("--heldout-generator", required=True)
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--learning-rate", type=float, default=2e-5)
    train_parser.add_argument("--seed", type=int, default=42)
    evaluate_parser = commands.add_parser("evaluate-fusion", help="Compare agent combinations.")
    evaluate_parser.add_argument("--predictions", type=Path, required=True)
    evaluate_parser.add_argument(
        "--output", type=Path, default=Path("reports/fusion_evaluation.json")
    )
    evaluate_parser.add_argument(
        "--model-output", type=Path, default=get_settings().fusion_model_path
    )
    evaluate_parser.add_argument("--database", type=Path, default=get_settings().database_path)
    evaluate_parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.command == "api":
        import uvicorn

        uvicorn.run("claimguard.api:app", host=args.host, port=args.port, reload=args.reload)
    elif args.command == "dashboard":
        from streamlit.web import cli as streamlit_cli

        sys.argv = ["streamlit", "run", "streamlit_app.py", "--server.port", str(args.port)]
        raise SystemExit(streamlit_cli.main())
    elif args.command == "validate-manifest":
        from claimguard.collect_predictions import validate_manifest

        print(validate_manifest(args.manifest))
    elif args.command == "collect":
        from claimguard.collect_predictions import collect_predictions

        collect_predictions(args.manifest, args.output, args.model, args.fusion_model)
    elif args.command == "train-detector":
        from claimguard.train_detector import train_detector

        train_detector(
            args.manifest,
            args.output_dir,
            args.base_model,
            args.heldout_generator,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.seed,
        )
    elif args.command == "evaluate-fusion":
        from claimguard.fusion import evaluate_fusion

        evaluate_fusion(
            args.predictions,
            args.output,
            args.model_output,
            args.database,
            args.seed,
        )
