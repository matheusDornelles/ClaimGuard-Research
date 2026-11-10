"""Fit, validate, and evaluate calibrated feature-fusion experiments."""

from __future__ import annotations

import itertools
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from claimguard.agents.decision import FEATURE_NAMES
from claimguard.metrics import binary_metrics
from claimguard.storage import ResearchStore

LOGGER = logging.getLogger(__name__)
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "metadata": FEATURE_NAMES[:7],
    "forensics": FEATURE_NAMES[7:11],
    "visual": FEATURE_NAMES[11:14],
    "ai_detector": FEATURE_NAMES[14:],
}


class PlattCalibratedClassifier:
    """A fitted base estimator plus sigmoid calibration on the held-out validation split."""

    def __init__(self, base: Pipeline, calibrator: LogisticRegression):
        self.base = base
        self.calibrator = calibrator

    def predict_proba(self, rows: list[list[float | int | None]] | np.ndarray) -> np.ndarray:
        raw = self.base.predict_proba(rows)[:, 1]
        logits = _logit(raw).reshape(-1, 1)
        positive = self.calibrator.predict_proba(logits)[:, 1]
        return np.column_stack((1.0 - positive, positive))


def evaluate_fusion(
    predictions_path: Path,
    output_path: Path,
    model_output: Path,
    database_path: Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare all available agent combinations on explicit train/validation/test splits."""
    data = pd.read_csv(predictions_path)
    required = {"binary_manipulated", "split", "dataset", "generator", "group_id"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Prediction CSV is missing columns: {', '.join(sorted(missing))}")
    _validate_splits(data)
    train = data[data["split"] == "train"].copy()
    validation = data[data["split"] == "validation"].copy()
    test = data[data["split"] == "test"].copy()
    _validate_binary_split(train, "train")
    _validate_binary_split(validation, "validation")
    _validate_binary_split(test, "test")
    _validate_cross_generator_holdout(train, test)

    available_groups = {
        group: tuple(column for column in columns if column in data.columns)
        for group, columns in FEATURE_GROUPS.items()
    }
    results: list[dict[str, Any]] = []
    full_artifact: dict[str, Any] | None = None

    group_names = tuple(FEATURE_GROUPS)
    for size in range(1, len(group_names) + 1):
        for selected in itertools.combinations(group_names, size):
            requested_features = tuple(
                feature for group in selected for feature in available_groups[group]
            )
            if not requested_features:
                results.append(_skipped_result(selected, "No feature columns are present."))
                continue
            result = _evaluate_subset(
                selected,
                requested_features,
                train,
                validation,
                test,
                seed,
            )
            artifact = result.pop("_artifact", None)
            results.append(result)
            if (
                set(selected) == set(group_names)
                and result.get("status") == "evaluated"
                and set(result.get("agents", [])) == set(group_names)
            ):
                full_artifact = artifact

    standalone_ai = _standalone_ai_metrics(test)
    summary = {
        "experiment_id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": "ai_generated_or_ai_edited_vs_real; not claim fraud",
        "dataset": ", ".join(sorted(data["dataset"].astype(str).unique())),
        "split_counts": {
            split: int((data["split"] == split).sum()) for split in ("train", "validation", "test")
        },
        "held_out_test_generators": sorted(test["generator"].astype(str).unique().tolist()),
        "standalone_ai_detector": standalone_ai,
        "agent_combinations": results,
        "evaluation_protocol": {
            "threshold": 0.5,
            "training": "Base logistic regression fit on training split only.",
            "calibration": "Sigmoid (Platt) calibration fit on validation split.",
            "test": "Final metrics reported once on the held-out test split.",
            "leakage_control": (
                "group_id values must be disjoint across train, validation, and test."
            ),
        },
    }

    if full_artifact:
        model_output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "feature_names": full_artifact["feature_names"],
                "classifier": full_artifact["classifier"],
                "model_name": f"calibrated-fusion-{summary['experiment_id'][:8]}",
                "target": summary["target"],
                "training_protocol": summary["evaluation_protocol"],
                "test_metrics": full_artifact["metrics"],
            },
            model_output,
        )
        summary["fusion_model_artifact"] = str(model_output.resolve())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if database_path:
        store = ResearchStore(database_path)
        store.save_experiment(
            experiment_id=summary["experiment_id"],
            name="agent-combination-fusion",
            dataset=summary["dataset"],
            metrics={
                "standalone_ai_detector": standalone_ai,
                "combinations": [
                    {key: value for key, value in item.items() if not key.startswith("_")}
                    for item in results
                ],
            },
            config=summary["evaluation_protocol"],
            created_at=datetime.fromisoformat(summary["created_at"]),
        )
    LOGGER.info("Wrote evaluation summary to %s", output_path)
    return summary


def _evaluate_subset(
    selected: tuple[str, ...],
    features: tuple[str, ...],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    seed: int,
) -> dict[str, Any]:
    train_features = train.loc[:, features]
    informative = [column for column in features if not train_features[column].isna().all()]
    if not informative:
        return _skipped_result(selected, "All candidate features are missing in training data.")
    used_agents = [group for group in selected if set(informative) & set(FEATURE_GROUPS[group])]

    y_train = train["binary_manipulated"].astype(int).to_numpy()
    y_validation = validation["binary_manipulated"].astype(int).to_numpy()
    y_test = test["binary_manipulated"].astype(int).to_numpy()
    base = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, random_state=seed)),
        ]
    )
    base.fit(train.loc[:, informative].to_numpy(dtype=float), y_train)
    validation_scores = base.predict_proba(validation.loc[:, informative].to_numpy(dtype=float))[
        :, 1
    ]
    calibrator = LogisticRegression(max_iter=1000, random_state=seed)
    calibrator.fit(_logit(validation_scores).reshape(-1, 1), y_validation)
    classifier = PlattCalibratedClassifier(base, calibrator)
    test_scores = classifier.predict_proba(test.loc[:, informative].to_numpy(dtype=float))[:, 1]
    metrics = binary_metrics(y_test, test_scores)

    per_generator = {}
    for generator in sorted(test["generator"].astype(str).unique()):
        mask = test["generator"].astype(str).to_numpy() == generator
        per_generator[generator] = binary_metrics(y_test[mask], test_scores[mask])

    return {
        "agents": used_agents,
        "requested_agents": list(selected),
        "status": "evaluated",
        "feature_names": informative,
        "metrics": metrics,
        "metrics_by_test_generator": per_generator,
        "_artifact": {
            "feature_names": informative,
            "classifier": classifier,
            "metrics": metrics,
        },
    }


def _standalone_ai_metrics(test: pd.DataFrame) -> dict[str, Any] | None:
    if not {"probability_ai_generated", "probability_ai_edited"} <= set(test.columns):
        return None
    columns = ["probability_ai_generated", "probability_ai_edited"]
    available = test[columns].notna().all(axis=1)
    if not available.any():
        return {"status": "unavailable", "reason": "No trained AI detector scores were collected."}
    filtered = test.loc[available]
    score = filtered[columns].sum(axis=1).to_numpy(dtype=float)
    metrics = binary_metrics(
        filtered["binary_manipulated"].astype(int).to_numpy(),
        score,
    )
    metrics["status"] = "evaluated"
    metrics["n_scored"] = int(available.sum())
    return metrics


def _validate_splits(data: pd.DataFrame) -> None:
    splits = set(data["split"].astype(str))
    if splits != {"train", "validation", "test"}:
        raise ValueError("Prediction data must contain exactly train, validation, and test splits.")
    group_sets = {
        split: set(data.loc[data["split"] == split, "group_id"].astype(str)) for split in splits
    }
    for first, second in itertools.combinations(sorted(splits), 2):
        overlap = group_sets[first] & group_sets[second]
        if overlap:
            raise ValueError(f"Data leakage: group_id overlap between {first} and {second}.")


def _validate_binary_split(data: pd.DataFrame, split_name: str) -> None:
    labels = set(data["binary_manipulated"].dropna().astype(int).unique())
    if labels != {0, 1}:
        raise ValueError(
            f"{split_name} split must contain both real (0) and manipulated (1) samples."
        )


def _validate_cross_generator_holdout(train: pd.DataFrame, test: pd.DataFrame) -> None:
    train_generators = set(train.loc[train["binary_manipulated"] == 1, "generator"].astype(str))
    test_generators = set(test.loc[test["binary_manipulated"] == 1, "generator"].astype(str))
    overlap = train_generators & test_generators
    if overlap:
        raise ValueError(
            "Cross-generator evaluation requires manipulated test generators absent from training; "
            f"overlap: {', '.join(sorted(overlap))}."
        )


def _skipped_result(agents: tuple[str, ...], reason: str) -> dict[str, Any]:
    return {"agents": list(agents), "status": "skipped", "reason": reason}


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def main() -> None:
    import argparse

    from claimguard.config import get_settings

    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/fusion_evaluation.json"))
    parser.add_argument("--model-output", type=Path, default=settings.fusion_model_path)
    parser.add_argument("--database", type=Path, default=settings.database_path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = evaluate_fusion(
        args.predictions,
        args.output,
        args.model_output,
        args.database,
        args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
