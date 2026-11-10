"""Calibrated fusion decision agent; deliberately abstains without a trained artifact."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from claimguard.schemas import AgentResult, AgentStatus

LOGGER = logging.getLogger(__name__)

FEATURE_NAMES = (
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
)


def build_feature_row(agents: dict[str, AgentResult]) -> dict[str, float | int | bool | None]:
    """Build the stable tabular fusion schema used by training and inference."""
    metadata = agents["metadata"].features
    forensic = agents["forensics"].features
    visual = agents["visual"].features
    ai = agents["ai_detector"].features
    return {
        "metadata_completeness": _number(metadata.get("metadata_completeness")),
        "camera_make_present": _indicator(metadata.get("camera_make_present")),
        "camera_model_present": _indicator(metadata.get("camera_model_present")),
        "timestamp_present": _indicator(metadata.get("timestamp_present")),
        "software_present": _indicator(metadata.get("software_present")),
        "gps_present": _indicator(metadata.get("gps_present")),
        "exif_tag_count": _number(metadata.get("exif_tag_count")),
        "ela_mean_abs_difference": _number(forensic.get("ela_mean_abs_difference")),
        "ela_difference_std": _number(forensic.get("ela_difference_std")),
        "jpeg_blockiness_contrast": _number(forensic.get("jpeg_blockiness_contrast")),
        "noise_residual_tile_cv": _number(forensic.get("noise_residual_tile_cv")),
        "tile_luminance_std": _number(visual.get("tile_luminance_std")),
        "tile_texture_variance_std": _number(visual.get("tile_texture_variance_std")),
        "edge_coverage_upper_quartile": _number(visual.get("edge_coverage_upper_quartile")),
        "probability_real": _number(ai.get("probability_real")),
        "probability_ai_generated": _number(ai.get("probability_ai_generated")),
        "probability_ai_edited": _number(ai.get("probability_ai_edited")),
    }


def create_decision(
    agents: dict[str, AgentResult],
    model_path: Path,
) -> tuple[float | None, str, str | None, list[str], AgentResult]:
    """Score only with a saved calibrated fusion model; otherwise abstain."""
    explanation = _explain(agents)
    if not model_path.is_file():
        decision_agent = AgentResult(
            agent="decision",
            status=AgentStatus.UNAVAILABLE,
            score_name="calibrated AI-manipulation probability",
            score_semantics=(
                "Unavailable until a fusion model is trained on labeled validation data."
            ),
            evidence=[
                "No trained/calibrated fusion artifact is configured; the system abstains "
                "rather than inventing a probability.",
                *explanation,
            ],
            features=build_feature_row(agents),
            model=None,
        )
        return None, "abstained_no_fusion_model", None, explanation, decision_agent

    artifact = _load_artifact(str(model_path))
    feature_names: list[str] = artifact["feature_names"]
    classifier = artifact["classifier"]
    features = build_feature_row(agents)
    feature_agents = {
        "metadata": set(FEATURE_NAMES[:7]),
        "forensics": set(FEATURE_NAMES[7:11]),
        "visual": set(FEATURE_NAMES[11:14]),
        "ai_detector": set(FEATURE_NAMES[14:]),
    }
    unavailable_agents = [
        name
        for name, agent_features in feature_agents.items()
        if set(feature_names) & agent_features and agents[name].status != AgentStatus.AVAILABLE
    ]
    if unavailable_agents:
        missing = ", ".join(unavailable_agents)
        decision_agent = AgentResult(
            agent="decision",
            status=AgentStatus.UNAVAILABLE,
            score_name="calibrated AI-manipulation probability",
            score_semantics=(
                "Unavailable because one or more trained-model input agents failed or abstained."
            ),
            evidence=[
                f"The trained fusion artifact requires unavailable agent(s): {missing}.",
                "The system abstains rather than imputing an entire failed agent as if it "
                "were an ordinary missing feature.",
                *explanation,
            ],
            features=features,
            model=str(artifact.get("model_name", model_path.name)),
        )
        return (
            None,
            "abstained_missing_required_agent",
            str(artifact.get("model_name", model_path.name)),
            explanation,
            decision_agent,
        )
    row = [[features.get(name) for name in feature_names]]
    probability = float(classifier.predict_proba(row)[0][1])
    model_name = str(artifact.get("model_name", model_path.name))
    decision_agent = AgentResult(
        agent="decision",
        status=AgentStatus.AVAILABLE,
        score=round(probability, 6),
        score_name="calibrated AI-manipulation probability",
        score_semantics=(
            "Calibrated on a held-out validation split for the artifact's documented dataset "
            "and operating domain; not a claim-fraud probability."
        ),
        confidence=None,
        evidence=[
            f"Fusion model: {model_name}.",
            "This estimate is only meaningful for data and domains represented by its "
            "documented evaluation.",
            *explanation,
        ],
        features=features,
        model=model_name,
    )
    return probability, "estimated_by_calibrated_model", model_name, explanation, decision_agent


def _explain(agents: dict[str, AgentResult]) -> list[str]:
    findings: list[str] = []
    for name in ("metadata", "forensics", "ai_detector", "visual"):
        result = agents[name]
        if result.status == AgentStatus.UNAVAILABLE:
            findings.append(
                f"{name.replace('_', ' ').title()} agent: unavailable; no inference was made."
            )
        elif result.status == AgentStatus.ERROR:
            findings.append(
                f"{name.replace('_', ' ').title()} agent: analysis error; its evidence is excluded."
            )
        else:
            findings.extend(result.evidence[:2])
    return findings


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    return float(value)


def _indicator(value: Any) -> int | None:
    return int(value) if isinstance(value, bool) else None


@lru_cache(maxsize=2)
def _load_artifact(model_path: str) -> dict[str, Any]:
    import joblib

    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict) or not {"feature_names", "classifier"} <= artifact.keys():
        raise ValueError(f"Fusion model artifact has an invalid schema: {model_path}")
    if not callable(getattr(artifact["classifier"], "predict_proba", None)):
        raise ValueError("Fusion model must provide calibrated predict_proba scores.")
    return artifact
