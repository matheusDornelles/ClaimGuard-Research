"""Dataset manifest validation with explicit labels and leakage checks."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

LABELS = ("real", "ai_generated", "ai_edited")
REQUIRED_COLUMNS = {"image_path", "label", "dataset", "generator", "split", "group_id"}
SPLITS = {"train", "validation", "test"}


@dataclass(frozen=True)
class DatasetRow:
    image_path: Path
    label: str
    dataset: str
    generator: str
    split: str
    group_id: str


def load_manifest(manifest_path: Path, require_files: bool = True) -> list[DatasetRow]:
    """Load a CSV manifest, validate labels/paths, and prevent group leakage."""
    manifest_path = manifest_path.resolve()
    with manifest_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"Dataset manifest is missing columns: {', '.join(sorted(missing))}")
        rows: list[DatasetRow] = []
        for line_number, raw in enumerate(reader, start=2):
            image_path = Path(raw["image_path"].strip())
            if not image_path.is_absolute():
                image_path = (manifest_path.parent / image_path).resolve()
            label = raw["label"].strip().lower()
            split = raw["split"].strip().lower()
            dataset = raw["dataset"].strip()
            generator = raw["generator"].strip()
            group_id = raw["group_id"].strip()
            if label not in LABELS:
                raise ValueError(f"Line {line_number}: label must be one of {LABELS}.")
            if split not in SPLITS:
                raise ValueError(f"Line {line_number}: split must be one of {sorted(SPLITS)}.")
            if not dataset or not generator or not group_id:
                raise ValueError(
                    f"Line {line_number}: dataset, generator, and group_id are required."
                )
            if require_files and not image_path.is_file():
                raise ValueError(f"Line {line_number}: image file does not exist: {image_path}")
            rows.append(DatasetRow(image_path, label, dataset, generator, split, group_id))

    if not rows:
        raise ValueError("Dataset manifest contains no image rows.")
    _validate_splits(rows)
    return rows


def _validate_splits(rows: list[DatasetRow]) -> None:
    split_groups: dict[str, set[str]] = {}
    for row in rows:
        split_groups.setdefault(row.split, set()).add(row.group_id)
    for split_a, groups_a in split_groups.items():
        for split_b, groups_b in split_groups.items():
            if split_a >= split_b:
                continue
            overlap = groups_a & groups_b
            if overlap:
                examples = ", ".join(sorted(overlap)[:5])
                raise ValueError(
                    f"Data leakage: group_id values occur in both {split_a!r} and "
                    f"{split_b!r}: {examples}"
                )
