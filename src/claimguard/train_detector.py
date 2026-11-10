"""Fine-tune a three-class Hugging Face image classifier on a grouped manifest."""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from claimguard.datasets import LABELS, DatasetRow, load_manifest
from claimguard.metrics import binary_metrics, multiclass_metrics

LOGGER = logging.getLogger(__name__)
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


def train_detector(
    manifest_path: Path,
    output_dir: Path,
    base_model: str,
    heldout_generator: str,
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    seed: int = 42,
) -> dict[str, Any]:
    """Train and evaluate a classifier; require explicit source/model holdout splits."""
    try:
        import torch
        from PIL import Image
        from transformers import (
            AutoImageProcessor,
            AutoModelForImageClassification,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as error:
        raise RuntimeError(
            'Training requires the optional ML dependencies: pip install -e ".[ml]"'
        ) from error

    rows = load_manifest(manifest_path)
    _validate_holdout(rows, heldout_generator)
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError(
            "epochs and batch_size must be positive; learning_rate must be greater than zero."
        )
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    processor = AutoImageProcessor.from_pretrained(base_model)
    model = AutoModelForImageClassification.from_pretrained(
        base_model,
        num_labels=len(LABELS),
        id2label={index: label for index, label in enumerate(LABELS)},
        label2id=LABEL_TO_ID,
        ignore_mismatched_sizes=True,
    )

    class ImageManifestDataset(torch.utils.data.Dataset):
        def __init__(self, examples: list[DatasetRow]):
            self.examples = examples

        def __len__(self) -> int:
            return len(self.examples)

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            example = self.examples[index]
            with Image.open(example.image_path) as source:
                image = source.convert("RGB")
                encoded = processor(images=image, return_tensors="pt")
            return {
                "pixel_values": encoded["pixel_values"].squeeze(0),
                "labels": torch.tensor(LABEL_TO_ID[example.label], dtype=torch.long),
            }

    splits = {
        split: [row for row in rows if row.split == split]
        for split in ("train", "validation", "test")
    }
    for split_name, examples in splits.items():
        counts = Counter(row.label for row in examples)
        if set(counts) != set(LABELS):
            raise ValueError(
                f"{split_name} split must contain all three classes; found {dict(counts)}."
            )
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        seed=seed,
        data_seed=seed,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        report_to=[],
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ImageManifestDataset(splits["train"]),
        eval_dataset=ImageManifestDataset(splits["validation"]),
        processing_class=processor,
        compute_metrics=_trainer_metrics,
    )
    trainer.train()
    test_output = trainer.predict(ImageManifestDataset(splits["test"]))
    probabilities = _softmax(test_output.predictions)
    true_ids = np.asarray(test_output.label_ids, dtype=int)
    binary_truth = (true_ids != LABEL_TO_ID["real"]).astype(int)
    binary_scores = (
        probabilities[:, LABEL_TO_ID["ai_generated"]] + probabilities[:, LABEL_TO_ID["ai_edited"]]
    )
    metrics = {
        "multiclass": multiclass_metrics(true_ids, probabilities, list(LABELS)),
        "binary_ai_manipulated_vs_real": binary_metrics(binary_truth, binary_scores),
        "by_test_generator": _metrics_by_generator(splits["test"], true_ids, probabilities),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    provenance = {
        "base_model": base_model,
        "label_mapping": LABEL_TO_ID,
        "heldout_generator": heldout_generator,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "train_samples": len(splits["train"]),
        "validation_samples": len(splits["validation"]),
        "test_samples": len(splits["test"]),
        "test_group_ids": sorted({row.group_id for row in splits["test"]}),
        "test_generators": sorted({row.generator for row in splits["test"]}),
        "metrics": metrics,
        "warning": "AI-manipulation classification does not establish claim fraud.",
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Saved model and held-out test report to %s", output_dir)
    return provenance


def _validate_holdout(rows: list[DatasetRow], heldout_generator: str) -> None:
    if not heldout_generator.strip():
        raise ValueError("A held-out generator name is required for cross-generator testing.")
    for row in rows:
        if row.label != "real" and row.generator == heldout_generator and row.split != "test":
            raise ValueError(
                f"Held-out generator {heldout_generator!r} may appear only in the test split."
            )
    test_generators = {row.generator for row in rows if row.label != "real" and row.split == "test"}
    if heldout_generator not in test_generators:
        raise ValueError(
            f"No manipulated test image uses the held-out generator {heldout_generator!r}."
        )
    train_generators = {
        row.generator for row in rows if row.label != "real" and row.split == "train"
    }
    if heldout_generator in train_generators:
        raise ValueError("The held-out generator overlaps the training data.")


def _trainer_metrics(prediction: Any) -> dict[str, float]:
    logits = (
        prediction.predictions[0]
        if isinstance(prediction.predictions, tuple)
        else prediction.predictions
    )
    probs = _softmax(logits)
    metrics = multiclass_metrics(prediction.label_ids, probs, list(LABELS))
    return {key: value for key, value in metrics.items() if isinstance(value, (int, float))}


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = np.asarray(logits, dtype=np.float64) - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _metrics_by_generator(
    rows: list[DatasetRow],
    true_ids: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for generator in sorted({row.generator for row in rows}):
        mask = np.array([row.generator == generator for row in rows])
        binary_truth = (true_ids[mask] != LABEL_TO_ID["real"]).astype(int)
        binary_scores = (
            probabilities[mask, LABEL_TO_ID["ai_generated"]]
            + probabilities[mask, LABEL_TO_ID["ai_edited"]]
        )
        results[generator] = binary_metrics(binary_truth, binary_scores)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", default="microsoft/resnet-50")
    parser.add_argument("--heldout-generator", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = train_detector(
        manifest_path=arguments.manifest,
        output_dir=arguments.output_dir,
        base_model=arguments.base_model,
        heldout_generator=arguments.heldout_generator,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        seed=arguments.seed,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
