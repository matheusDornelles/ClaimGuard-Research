"""Optional Transformer image classifier, configured only with an explicit model."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from io import BytesIO
from typing import Any

from PIL import Image

from claimguard.schemas import AgentResult, AgentStatus

LOGGER = logging.getLogger(__name__)
SUPPORTED_CLASSES = ("real", "ai_generated", "ai_edited")


def analyze_ai_image(image_bytes: bytes, model_id: str | None) -> AgentResult:
    """Infer three image classes if a configured Hugging Face model is available."""
    model_id = model_id or os.environ.get("CLAIMGUARD_IMAGE_MODEL")
    if not model_id:
        return AgentResult(
            agent="ai_detector",
            status=AgentStatus.UNAVAILABLE,
            score_name="AI/manipulation class probabilities",
            score_semantics="Unavailable until a three-class model is trained and configured.",
            evidence=[
                "No trained image-classification model is configured.",
                "No image-origin classification is inferred from metadata or "
                "handcrafted forensic features.",
            ],
        )

    try:
        processor, model, torch = _load_model(model_id)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.inference_mode():
            logits = model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)[0].cpu().tolist()
    except ImportError as error:
        LOGGER.exception("AI detector dependencies are missing")
        return AgentResult(
            agent="ai_detector",
            status=AgentStatus.UNAVAILABLE,
            score_name="AI/manipulation class probabilities",
            score_semantics=(
                "Inference was not run because optional ML dependencies are unavailable."
            ),
            error=str(error),
            evidence=["Install the optional ML dependencies and configure a trained model."],
        )

    labels = _class_labels(model.config.id2label, len(probabilities))
    class_probabilities = {
        label: float(probability) for label, probability in zip(labels, probabilities, strict=True)
    }
    predicted_label = max(class_probabilities, key=class_probabilities.get)
    return AgentResult(
        agent="ai_detector",
        status=AgentStatus.AVAILABLE,
        score=round(class_probabilities["ai_generated"] + class_probabilities["ai_edited"], 6),
        score_name="model synthetic/manipulated class probability",
        score_semantics=(
            "Softmax output from the configured model; calibrated only if established by "
            "its training protocol."
        ),
        confidence=round(float(class_probabilities[predicted_label]), 6),
        evidence=[f"Configured classifier's highest-probability class: {predicted_label}."],
        features={
            f"probability_{label}": round(value, 6) for label, value in class_probabilities.items()
        },
        model=model_id,
    )


@lru_cache(maxsize=2)
def _load_model(model_id: str) -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification
    except ImportError as error:
        raise ImportError("AI classification requires the optional 'ml' dependencies.") from error

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id)
    labels = _class_labels(model.config.id2label, model.config.num_labels)
    if set(labels) != set(SUPPORTED_CLASSES):
        raise ValueError(
            f"Model {model_id!r} must expose exactly these classes: {', '.join(SUPPORTED_CLASSES)}."
        )
    model.eval()
    return processor, model, torch


def _class_labels(id2label: dict[Any, str], num_labels: int) -> list[str]:
    normalized = {int(index): _normalize_label(label) for index, label in id2label.items()}
    labels = [normalized.get(index, "") for index in range(num_labels)]
    if set(labels) != set(SUPPORTED_CLASSES):
        raise ValueError(
            "The configured image classifier must map model outputs to "
            "'real', 'ai_generated', and 'ai_edited'."
        )
    return labels


def _normalize_label(label: str) -> str:
    normalized = label.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "fake": "ai_generated",
        "synthetic": "ai_generated",
        "generated": "ai_generated",
        "edited": "ai_edited",
        "manipulated": "ai_edited",
        "authentic": "real",
    }
    return aliases.get(normalized, normalized)
