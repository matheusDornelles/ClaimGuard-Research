"""Shared application service used by FastAPI and Streamlit."""

from __future__ import annotations

from claimguard.config import Settings
from claimguard.graph import analyze_image
from claimguard.schemas import AnalysisReport
from claimguard.storage import ResearchStore


def analyze_and_store(
    image_bytes: bytes,
    filename: str,
    media_type: str,
    settings: Settings,
    store: ResearchStore | None = None,
) -> AnalysisReport:
    """Analyze bytes, then persist only the resulting report and image digest."""
    report = analyze_image(
        image_bytes=image_bytes,
        filename=filename,
        media_type=media_type,
        model_path=settings.model_path,
        fusion_model_path=settings.fusion_model_path,
    )
    (store or ResearchStore(settings.database_path)).save_analysis(report)
    return report
