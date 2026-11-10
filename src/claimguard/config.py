"""Environment-backed configuration for local research runs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    database_path: Path
    model_path: str | None
    fusion_model_path: Path
    max_upload_bytes: int
    allowed_mime_types: tuple[str, ...]


def get_settings() -> Settings:
    """Read settings from environment variables, with safe local defaults."""
    default_data_dir = PROJECT_ROOT / "data"
    db_path = Path(os.environ.get("CLAIMGUARD_DB", default_data_dir / "claimguard.sqlite3"))
    fusion_path = Path(
        os.environ.get("CLAIMGUARD_FUSION_MODEL", PROJECT_ROOT / "artifacts" / "fusion.joblib")
    )
    max_upload = int(os.environ.get("CLAIMGUARD_MAX_UPLOAD_BYTES", 25 * 1024 * 1024))
    if max_upload <= 0:
        raise ValueError("CLAIMGUARD_MAX_UPLOAD_BYTES must be positive")
    model_path = os.environ.get("CLAIMGUARD_IMAGE_MODEL")
    return Settings(
        database_path=db_path,
        model_path=model_path,
        fusion_model_path=fusion_path,
        max_upload_bytes=max_upload,
        allowed_mime_types=("image/jpeg", "image/png", "image/webp", "image/tiff", "image/bmp"),
    )
