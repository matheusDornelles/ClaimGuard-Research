"""Binary and multiclass metrics used by reproducible evaluation scripts."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(
    y_true: list[int] | np.ndarray,
    probabilities: list[float] | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute fixed-threshold metrics and ROC-AUC for manipulated vs real."""
    labels = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    predictions = (scores >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    auc = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": auc,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else None,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else None,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "n": int(len(labels)),
    }


def multiclass_metrics(
    y_true: list[int] | np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    """Compute multiclass macro metrics plus one-vs-rest macro ROC-AUC."""
    truth = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    predictions = probs.argmax(axis=1)
    try:
        auc = float(roc_auc_score(truth, probs, multi_class="ovr", average="macro"))
    except ValueError:
        auc = None
    matrix = confusion_matrix(truth, predictions, labels=list(range(len(labels))))
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "precision_macro": float(
            precision_score(truth, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(recall_score(truth, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(truth, predictions, average="macro", zero_division=0)),
        "roc_auc_ovr_macro": auc,
        "labels": labels,
        "confusion_matrix": matrix.tolist(),
        "n": int(len(truth)),
    }
