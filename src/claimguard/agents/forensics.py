"""Exploratory compression and noise indicators, not a calibrated detector."""

from __future__ import annotations

import cv2
import numpy as np

from claimguard.schemas import AgentResult, AgentStatus


def analyze_forensics(rgb: np.ndarray, file_format: str) -> AgentResult:
    """Compute ELA-like, JPEG block, and noise-residual descriptive indicators."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.size == 0:
        raise ValueError("Forensics expects a non-empty RGB image array.")

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    noise = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    tile_variances = _tile_values(noise, operation="variance")
    noise_variation = _coefficient_of_variation(tile_variances)

    jpeg_input = file_format.lower() in {"jpeg", "jpg"}
    ela_mean: float | None = None
    ela_std: float | None = None
    blockiness: float | None = None

    if jpeg_input:
        encoded_ok, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        )
        if encoded_ok:
            recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            recompressed_rgb = cv2.cvtColor(recompressed, cv2.COLOR_BGR2RGB)
            difference = cv2.absdiff(rgb, recompressed_rgb).astype(np.float32)
            ela_mean = float(np.mean(difference))
            ela_std = float(np.std(difference))
            gray_float = gray.astype(np.float32)
            vertical = np.abs(np.diff(gray_float, axis=1))
            horizontal = np.abs(np.diff(gray_float, axis=0))
            v_boundary = _mean_or_zero(vertical[:, 7::8])
            v_interior = _mean_or_zero(
                np.delete(vertical, np.arange(7, vertical.shape[1], 8), axis=1)
            )
            h_boundary = _mean_or_zero(horizontal[7::8, :])
            h_interior = _mean_or_zero(
                np.delete(horizontal, np.arange(7, horizontal.shape[0], 8), axis=0)
            )
            blockiness = float(
                max(0.0, (v_boundary + h_boundary) / 2 - (v_interior + h_interior) / 2)
            )

    features: dict[str, float | int | bool | str | None] = {
        "jpeg_input": jpeg_input,
        "ela_mean_abs_difference": round(ela_mean, 5) if ela_mean is not None else None,
        "ela_difference_std": round(ela_std, 5) if ela_std is not None else None,
        "jpeg_blockiness_contrast": round(blockiness, 5) if blockiness is not None else None,
        "noise_residual_tile_cv": round(noise_variation, 5),
        "noise_residual_median_tile_variance": round(float(np.median(tile_variances)), 5),
    }
    evidence = [
        "These compression and noise summaries are exploratory image cues, not "
        "manipulation probabilities.",
        "ELA values are only computed for JPEG inputs and are sensitive to image content, "
        "prior edits, and recompression.",
        f"Noise-residual variation across image tiles: {noise_variation:.3f}.",
    ]
    if jpeg_input:
        evidence.append("JPEG recompression-difference and 8×8 boundary summaries were computed.")
    else:
        evidence.append(
            "ELA and JPEG block summaries are unavailable because the source is not JPEG."
        )

    return AgentResult(
        agent="forensics",
        status=AgentStatus.AVAILABLE,
        score=None,
        score_name="forensic cues",
        score_semantics=(
            "Raw descriptive features; no calibrated manipulation likelihood is estimated."
        ),
        evidence=evidence,
        features=features,
        model="OpenCV Laplacian residual and JPEG recompression summaries",
    )


def _tile_values(image: np.ndarray, operation: str, grid: int = 4) -> list[float]:
    height, width = image.shape[:2]
    values: list[float] = []
    for row in range(grid):
        for column in range(grid):
            y0, y1 = row * height // grid, (row + 1) * height // grid
            x0, x1 = column * width // grid, (column + 1) * width // grid
            tile = image[y0:y1, x0:x1]
            if tile.size:
                values.append(float(np.var(tile) if operation == "variance" else np.mean(tile)))
    return values or [0.0]


def _coefficient_of_variation(values: list[float]) -> float:
    mean = float(np.mean(values))
    if mean <= 1e-8:
        return 0.0
    return float(np.std(values) / mean)


def _mean_or_zero(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else 0.0
