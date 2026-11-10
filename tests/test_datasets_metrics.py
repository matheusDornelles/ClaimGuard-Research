from __future__ import annotations

import csv
from pathlib import Path

import pytest

from claimguard.collect_predictions import collect_predictions
from claimguard.datasets import load_manifest
from claimguard.metrics import binary_metrics


def _write_manifest(path: Path, group_test: str = "g-test") -> None:
    rows = [
        ("real.jpg", "real", "fixture", "real", "train", "g-train"),
        ("fake.jpg", "ai_generated", "fixture", "model-a", "validation", "g-validation"),
        ("test.jpg", "ai_edited", "fixture", "model-b", "test", group_test),
    ]
    for filename, *_ in rows:
        (path.parent / filename).write_bytes(b"fixture")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "label", "dataset", "generator", "split", "group_id"])
        writer.writerows(rows)


def test_manifest_loads_and_checks_grouped_splits(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest)
    rows = load_manifest(manifest)
    assert len(rows) == 3
    assert rows[-1].generator == "model-b"


def test_manifest_rejects_group_leakage(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, group_test="g-train")
    with pytest.raises(ValueError, match="Data leakage"):
        load_manifest(manifest)


def test_binary_metrics_return_all_requested_rates() -> None:
    metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9])
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["false_positive_rate"] == 0.0
    assert metrics["false_negative_rate"] == 0.0


def test_collect_predictions_runs_agents_for_manifest(tmp_path: Path, png_bytes: bytes) -> None:
    manifest = tmp_path / "manifest.csv"
    records = [
        ("real.png", "real", "real-camera", "train", "real-group"),
        ("generated.png", "ai_generated", "generator-a", "validation", "generated-group"),
        ("edited.png", "ai_edited", "generator-b", "test", "edited-group"),
    ]
    for filename, *_ in records:
        (tmp_path / filename).write_bytes(png_bytes)
    with manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "label", "dataset", "generator", "split", "group_id"])
        writer.writerows(
            (filename, label, "fixture", generator, split, group_id)
            for filename, label, generator, split, group_id in records
        )

    output = tmp_path / "predictions.csv"
    count = collect_predictions(
        manifest_path=manifest,
        output_path=output,
        model_path=None,
        fusion_model_path=tmp_path / "missing.joblib",
    )

    assert count == len(records)
    with output.open(newline="", encoding="utf-8") as file:
        result_rows = list(csv.DictReader(file))
    assert len(result_rows) == len(records)
    assert result_rows[0]["binary_manipulated"] == "0"
    assert result_rows[1]["binary_manipulated"] == "1"
    assert result_rows[0]["noise_residual_tile_cv"] != ""
