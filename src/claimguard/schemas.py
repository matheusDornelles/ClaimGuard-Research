"""Typed result models shared by analysis agents, API, and dashboards."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AgentStatus(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


class AgentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent: str
    status: AgentStatus
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    score_name: str | None = None
    score_semantics: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    features: dict[str, float | int | bool | str | None] = Field(default_factory=dict)
    model: str | None = None
    error: str | None = None


class AnalysisReport(BaseModel):
    analysis_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    filename: str
    sha256: str
    image_width: int
    image_height: int
    agents: dict[str, AgentResult]
    manipulation_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    probability_target: str = "AI-generated or AI-edited image, not claim fraud"
    decision_status: str
    decision_model: str | None = None
    explanation: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class ExperimentRecord(BaseModel):
    experiment_id: str
    created_at: datetime
    name: str
    dataset: str
    metrics: dict[str, Any]
    config: dict[str, Any]


def current_time() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)
