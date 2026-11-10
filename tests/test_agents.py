from __future__ import annotations

import io
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression

import claimguard.imaging as imaging
from claimguard.agents.ai_detector import analyze_ai_image
from claimguard.agents.decision import create_decision
from claimguard.agents.forensics import analyze_forensics
from claimguard.agents.metadata import analyze_metadata
from claimguard.agents.visual import analyze_visual
from claimguard.imaging import decode_image, image_array
from claimguard.schemas import AgentStatus


def test_metadata_absence_is_not_scored_as_fraud(png_bytes: bytes) -> None:
    result = analyze_metadata(png_bytes)
    assert result.status == AgentStatus.AVAILABLE
    assert result.score_name == "metadata completeness"
    assert result.score_semantics and "not a trust or fraud probability" in result.score_semantics
    assert "Missing or stripped metadata" in result.evidence[1]
    assert "gps_latitude" not in result.features


def test_forensics_reports_raw_cues_not_probability(png_bytes: bytes, jpeg_bytes: bytes) -> None:
    png = decode_image(png_bytes)
    jpeg = decode_image(jpeg_bytes)
    png_result = analyze_forensics(image_array(png.image), "png")
    jpeg_result = analyze_forensics(image_array(jpeg.image), "jpeg")

    assert png_result.score is None
    assert png_result.features["ela_mean_abs_difference"] is None
    assert jpeg_result.score is None
    assert jpeg_result.features["ela_mean_abs_difference"] is not None
    assert jpeg_result.features["jpeg_blockiness_contrast"] is not None


def test_forensics_handles_minimal_jpeg() -> None:
    image = Image.new("RGB", (1, 1), color="white")
    stream = io.BytesIO()
    image.save(stream, format="JPEG")
    result = analyze_forensics(image_array(image), "jpeg")
    assert all(value is None or np.isfinite(value) for value in result.features.values())


def test_decode_rejects_images_above_pixel_limit(monkeypatch) -> None:
    monkeypatch.setattr(imaging, "MAX_IMAGE_PIXELS", 32)
    image = Image.new("RGB", (8, 8), color="white")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    try:
        imaging.decode_image(stream.getvalue())
    except ValueError as error:
        assert "pixel safety limit" in str(error)
    else:
        raise AssertionError("Expected the oversized image to be rejected.")


def test_visual_agent_does_not_infer_shadow_or_reflection_truth(png_bytes: bytes) -> None:
    decoded = decode_image(png_bytes)
    result = analyze_visual(image_array(decoded.image))
    assert result.score is None
    assert result.features["reflection_consistency"] is None
    assert result.features["shadow_direction_consistency"] is None


def test_ai_detector_abstains_without_explicit_model(png_bytes: bytes) -> None:
    result = analyze_ai_image(png_bytes, None)
    assert result.status == AgentStatus.UNAVAILABLE
    assert result.score is None
    assert "No image-origin classification is inferred" in result.evidence[1]


def test_decision_abstains_without_calibrated_artifact(png_bytes: bytes, tmp_path: Path) -> None:
    metadata = analyze_metadata(png_bytes)
    decoded = decode_image(png_bytes)
    agents = {
        "metadata": metadata,
        "forensics": analyze_forensics(image_array(decoded.image), "png"),
        "ai_detector": analyze_ai_image(png_bytes, None),
        "visual": analyze_visual(image_array(decoded.image)),
    }

    probability, status, model, _, decision_agent = create_decision(
        agents,
        tmp_path / "not-trained.joblib",
    )
    assert probability is None
    assert status == "abstained_no_fusion_model"
    assert model is None
    assert decision_agent.status == AgentStatus.UNAVAILABLE


def test_decision_abstains_when_a_required_agent_is_unavailable(
    png_bytes: bytes,
    tmp_path: Path,
) -> None:
    decoded = decode_image(png_bytes)
    agents = {
        "metadata": analyze_metadata(png_bytes),
        "forensics": analyze_forensics(image_array(decoded.image), "png"),
        "ai_detector": analyze_ai_image(png_bytes, None),
        "visual": analyze_visual(image_array(decoded.image)),
    }
    estimator = LogisticRegression().fit([[0.0], [0.2], [0.8], [1.0]], [0, 0, 1, 1])
    model_path = tmp_path / "requires-ai-model.joblib"
    joblib.dump(
        {
            "feature_names": ["probability_ai_generated"],
            "classifier": estimator,
            "model_name": "fixture",
        },
        model_path,
    )

    probability, status, _, _, decision_agent = create_decision(agents, model_path)

    assert probability is None
    assert status == "abstained_missing_required_agent"
    assert decision_agent.status == AgentStatus.UNAVAILABLE
    assert "ai_detector" in decision_agent.evidence[0]
