from __future__ import annotations

from pathlib import Path

from claimguard.graph import analyze_image
from claimguard.schemas import AgentStatus
from claimguard.storage import ResearchStore


def test_graph_produces_all_agent_outputs_and_abstains(png_bytes: bytes, tmp_path: Path) -> None:
    report = analyze_image(
        image_bytes=png_bytes,
        filename="benchmark.png",
        media_type="image/png",
        model_path=None,
        fusion_model_path=tmp_path / "missing.joblib",
    )

    assert set(report.agents) == {
        "metadata",
        "forensics",
        "ai_detector",
        "visual",
        "decision",
    }
    assert report.manipulation_probability is None
    assert report.probability_target == "AI-generated or AI-edited image, not claim fraud"
    assert report.agents["ai_detector"].status == AgentStatus.UNAVAILABLE
    assert report.decision_status == "abstained_no_fusion_model"


def test_storage_persists_report_without_image_bytes(png_bytes: bytes, tmp_path: Path) -> None:
    report = analyze_image(
        image_bytes=png_bytes,
        filename="benchmark.png",
        media_type="image/png",
        model_path=None,
        fusion_model_path=tmp_path / "missing.joblib",
    )
    database = tmp_path / "research.sqlite3"
    store = ResearchStore(database)
    store.save_analysis(report)

    stored = store.get_analysis(report.analysis_id)
    assert stored is not None
    assert stored.sha256 == report.sha256
    assert store.list_analyses(10)[0].analysis_id == report.analysis_id
    assert png_bytes not in database.read_bytes()
