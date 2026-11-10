from __future__ import annotations

import csv
from pathlib import Path

import joblib

from claimguard.fusion import evaluate_fusion
from claimguard.storage import ResearchStore


def _write_predictions(path: Path) -> None:
    fields = [
        "sample_id",
        "binary_manipulated",
        "label",
        "dataset",
        "generator",
        "split",
        "group_id",
        "metadata_completeness",
        "camera_make_present",
        "camera_model_present",
        "timestamp_present",
        "software_present",
        "gps_present",
        "exif_tag_count",
        "ela_mean_abs_difference",
        "ela_difference_std",
        "jpeg_blockiness_contrast",
        "noise_residual_tile_cv",
        "tile_luminance_std",
        "tile_texture_variance_std",
        "edge_coverage_upper_quartile",
        "probability_real",
        "probability_ai_generated",
        "probability_ai_edited",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        index = 0
        for split, generator, count in (
            ("train", "model-train", 60),
            ("validation", "model-train", 30),
            ("test", "model-heldout", 30),
        ):
            for sample in range(count):
                label = sample % 2
                signal = float(label)
                writer.writerow(
                    {
                        "sample_id": f"sample-{index}",
                        "binary_manipulated": label,
                        "label": "ai_generated" if label else "real",
                        "dataset": "synthetic-fixture",
                        "generator": generator if label else "real-camera",
                        "split": split,
                        "group_id": f"group-{index}",
                        "metadata_completeness": 0.8 - signal * 0.5,
                        "camera_make_present": 1 - label,
                        "camera_model_present": 1 - label,
                        "timestamp_present": 1 - label,
                        "software_present": label,
                        "gps_present": 1 - label,
                        "exif_tag_count": 8 - label * 6,
                        "ela_mean_abs_difference": 0.1 + signal * 0.8,
                        "ela_difference_std": 0.2 + signal * 0.6,
                        "jpeg_blockiness_contrast": 0.1 + signal * 0.7,
                        "noise_residual_tile_cv": 0.2 + signal * 0.5,
                        "tile_luminance_std": 0.2 + signal * 0.4,
                        "tile_texture_variance_std": 0.1 + signal * 0.6,
                        "edge_coverage_upper_quartile": 0.7 - signal * 0.2,
                        "probability_real": 0.8 - signal * 0.7,
                        "probability_ai_generated": 0.1 + signal * 0.7,
                        "probability_ai_edited": 0.1 + signal * 0.1,
                    }
                )
                index += 1


def test_fusion_compares_combinations_calibrates_and_saves(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.csv"
    output = tmp_path / "metrics.json"
    model_path = tmp_path / "artifacts" / "fusion.joblib"
    database = tmp_path / "experiments.sqlite3"
    _write_predictions(predictions)

    summary = evaluate_fusion(predictions, output, model_path, database)

    assert len(summary["agent_combinations"]) == 15
    full = next(
        result
        for result in summary["agent_combinations"]
        if result.get("agents") == ["metadata", "forensics", "visual", "ai_detector"]
    )
    assert full["status"] == "evaluated"
    assert full["metrics"]["accuracy"] >= 0.9
    assert summary["held_out_test_generators"] == ["model-heldout", "real-camera"]
    artifact = joblib.load(model_path)
    assert artifact["classifier"].predict_proba([[0.5] * len(artifact["feature_names"])]).shape == (
        1,
        2,
    )
    assert len(ResearchStore(database).list_experiments()) == 1


def test_fusion_rejects_seen_test_generator(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.csv"
    _write_predictions(predictions)
    text = predictions.read_text(encoding="utf-8").replace("model-heldout", "model-train")
    predictions.write_text(text, encoding="utf-8")
    try:
        evaluate_fusion(
            predictions,
            tmp_path / "metrics.json",
            tmp_path / "fusion.joblib",
        )
    except ValueError as error:
        assert "test generators absent from training" in str(error)
    else:
        raise AssertionError("Expected generator leakage to be rejected.")
