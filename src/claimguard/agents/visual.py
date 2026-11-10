"""Interpretable low-level visual cues; semantic claims require separate models."""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

from claimguard.schemas import AgentResult, AgentStatus


def analyze_visual(rgb: np.ndarray) -> AgentResult:
    """Summarize tile-level illumination and texture variation."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.size == 0:
        raise ValueError("Visual analysis expects a non-empty RGB image array.")

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    luminance = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32) / 255.0
    illumination_means = _tile_stat(luminance, np.mean)
    texture_variances = _tile_stat(gray, lambda tile: np.var(cv2.Laplacian(tile, cv2.CV_32F)))

    lighting_spread = float(np.std(illumination_means))
    texture_spread = float(np.std(texture_variances))
    gradients_x = cv2.Sobel(luminance, cv2.CV_32F, 1, 0, ksize=3)
    gradients_y = cv2.Sobel(luminance, cv2.CV_32F, 0, 1, ksize=3)
    gradient_energy = np.sqrt(gradients_x**2 + gradients_y**2)
    edge_coverage = float(np.mean(gradient_energy > np.percentile(gradient_energy, 75)))

    features: dict[str, float | int | bool | str | None] = {
        "tile_luminance_std": round(lighting_spread, 5),
        "tile_texture_variance_std": round(texture_spread, 5),
        "edge_coverage_upper_quartile": round(edge_coverage, 5),
        "reflection_consistency": None,
        "shadow_direction_consistency": None,
    }
    evidence = [
        f"Tile-level luminance variation: {lighting_spread:.3f}; "
        f"image texture variation: {texture_spread:.3f}.",
        "Lighting and texture summaries depend strongly on scene content and are not "
        "authenticity judgments.",
        "Generic shadow and reflection consistency are not scored: they require "
        "object/scene context and validated annotations.",
    ]
    return AgentResult(
        agent="visual",
        status=AgentStatus.AVAILABLE,
        score=None,
        score_name="visual consistency cues",
        score_semantics=(
            "Low-level descriptive features; no calibrated authenticity score is estimated."
        ),
        evidence=evidence,
        features=features,
        model="OpenCV luminance, Sobel gradients, and Laplacian texture summaries",
    )


def _tile_stat(
    image: np.ndarray,
    statistic: Callable[[np.ndarray], float | np.generic],
    grid: int = 4,
) -> list[float]:
    height, width = image.shape[:2]
    values: list[float] = []
    for row in range(grid):
        for column in range(grid):
            tile = image[
                row * height // grid : (row + 1) * height // grid,
                column * width // grid : (column + 1) * width // grid,
            ]
            if tile.size:
                values.append(float(statistic(tile)))
    return values or [0.0]
