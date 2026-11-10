"""SQLite persistence for experiment outputs; uploaded image bytes are never stored."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from claimguard.schemas import AnalysisReport, ExperimentRecord


class ResearchStore:
    """Small SQLite store for analysis reports and benchmark experiment summaries."""

    def __init__(self, database_path: Path):
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS analyses (
                    analysis_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    image_sha256 TEXT NOT NULL,
                    decision_status TEXT NOT NULL,
                    manipulation_probability REAL,
                    report_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    name TEXT NOT NULL,
                    dataset TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    config_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS analyses_created_idx ON analyses(created_at)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS experiments_created_idx ON experiments(created_at)"
            )

    def save_analysis(self, report: AnalysisReport) -> None:
        payload = report.model_dump_json()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO analyses
                (
                    analysis_id, created_at, image_sha256, decision_status,
                    manipulation_probability, report_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    report.analysis_id,
                    report.created_at.isoformat(),
                    report.sha256,
                    report.decision_status,
                    report.manipulation_probability,
                    payload,
                ),
            )

    def get_analysis(self, analysis_id: str) -> AnalysisReport | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT report_json FROM analyses WHERE analysis_id = ?",
                (analysis_id,),
            ).fetchone()
        return AnalysisReport.model_validate_json(row["report_json"]) if row else None

    def list_analyses(self, limit: int = 50) -> list[AnalysisReport]:
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT report_json FROM analyses ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [AnalysisReport.model_validate_json(row["report_json"]) for row in rows]

    def save_experiment(
        self,
        experiment_id: str,
        name: str,
        dataset: str,
        metrics: dict[str, Any],
        config: dict[str, Any],
        created_at: datetime,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO experiments
                (experiment_id, created_at, name, dataset, metrics_json, config_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    created_at.isoformat(),
                    name,
                    dataset,
                    json.dumps(metrics, sort_keys=True),
                    json.dumps(config, sort_keys=True),
                ),
            )

    def list_experiments(self, limit: int = 50) -> list[ExperimentRecord]:
        if not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT experiment_id, created_at, name, dataset, metrics_json, config_json
                FROM experiments ORDER BY created_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            ExperimentRecord(
                experiment_id=row["experiment_id"],
                created_at=datetime.fromisoformat(row["created_at"]),
                name=row["name"],
                dataset=row["dataset"],
                metrics=json.loads(row["metrics_json"]),
                config=json.loads(row["config_json"]),
            )
            for row in rows
        ]

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
