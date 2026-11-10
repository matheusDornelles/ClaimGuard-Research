from __future__ import annotations

from dataclasses import replace

from fastapi.testclient import TestClient

from claimguard import api
from claimguard.storage import ResearchStore


def test_api_health_upload_and_analysis_history(
    png_bytes: bytes,
    tmp_path,
    monkeypatch,
) -> None:
    settings = replace(
        api.settings,
        database_path=tmp_path / "api.sqlite3",
        model_path=None,
        fusion_model_path=tmp_path / "missing.joblib",
        max_upload_bytes=1024 * 1024,
    )
    store = ResearchStore(settings.database_path)
    monkeypatch.setattr(api, "settings", settings)
    monkeypatch.setattr(api, "store", store)

    with TestClient(api.app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        rejected = client.post(
            "/api/analyze",
            files={"file": ("unsupported.svg", b"<svg/>", "image/svg+xml")},
        )
        assert rejected.status_code == 415

        response = client.post(
            "/api/analyze",
            files={"file": ("fixture.png", png_bytes, "image/png")},
        )
        assert response.status_code == 200
        report = response.json()
        assert report["manipulation_probability"] is None
        assert report["decision_status"] == "abstained_no_fusion_model"

        history = client.get("/api/analyses")
        assert history.status_code == 200
        assert history.json()[0]["analysis_id"] == report["analysis_id"]
