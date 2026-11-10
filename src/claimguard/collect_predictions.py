"""Run metadata/forensic/visual/AI agents on a labeled manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from claimguard.agents.decision import FEATURE_NAMES, build_feature_row
from claimguard.config import get_settings
from claimguard.datasets import LABELS, load_manifest
from claimguard.graph import analyze_image

LOGGER = logging.getLogger(__name__)


def collect_predictions(
    manifest_path: Path,
    output_path: Path,
    model_path: str | None,
    fusion_model_path: Path,
) -> int:
    rows = load_manifest(manifest_path)
    columns = [
        "sample_id",
        "image_path",
        "label",
        "binary_manipulated",
        "dataset",
        "generator",
        "split",
        "group_id",
        *FEATURE_NAMES,
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            image_bytes = row.image_path.read_bytes()
            report = analyze_image(
                image_bytes=image_bytes,
                filename=row.image_path.name,
                media_type=_mime_type(row.image_path),
                model_path=model_path,
                fusion_model_path=fusion_model_path,
            )
            features = build_feature_row(report.agents)
            writer.writerow(
                {
                    "sample_id": hashlib.sha256(
                        f"{row.group_id}:{row.image_path}".encode()
                    ).hexdigest(),
                    "image_path": str(row.image_path),
                    "label": row.label,
                    "binary_manipulated": int(row.label != "real"),
                    "dataset": row.dataset,
                    "generator": row.generator,
                    "split": row.split,
                    "group_id": row.group_id,
                    **features,
                }
            )
            if index % 100 == 0:
                LOGGER.info("Analyzed %d/%d images", index, len(rows))
    LOGGER.info("Wrote %d prediction rows to %s", len(rows), output_path)
    return len(rows)


def validate_manifest(path: Path) -> dict[str, Any]:
    rows = load_manifest(path)
    return {
        "rows": len(rows),
        "datasets": sorted({row.dataset for row in rows}),
        "generators": sorted({row.generator for row in rows}),
        "split_counts": {
            split: sum(row.split == split for row in rows)
            for split in ("train", "validation", "test")
        },
        "class_counts": {label: sum(row.label == label for row in rows) for label in LABELS},
        "groups": len({row.group_id for row in rows}),
    }


def _mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".bmp": "image/bmp",
    }.get(suffix, "application/octet-stream")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=get_settings().model_path)
    parser.add_argument("--fusion-model", type=Path, default=get_settings().fusion_model_path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--summary-json", type=Path)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if arguments.validate_only:
        summary = validate_manifest(arguments.manifest)
        text = json.dumps(summary, indent=2)
        if arguments.summary_json:
            arguments.summary_json.parent.mkdir(parents=True, exist_ok=True)
            arguments.summary_json.write_text(text + "\n", encoding="utf-8")
        print(text)
        return
    collect_predictions(
        arguments.manifest,
        arguments.output,
        arguments.model,
        arguments.fusion_model,
    )


if __name__ == "__main__":
    main()
