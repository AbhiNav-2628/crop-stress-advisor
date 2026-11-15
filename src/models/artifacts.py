"""Saving and loading model artifacts as JSON.

JSON is used instead of pickle on purpose: it is human-readable, cannot run
arbitrary code when loaded, and does not break when scikit-learn is upgraded.
"""

import json
from pathlib import Path

import numpy as np

from src.features.feature_engineering import FEATURE_ORDER

REQUIRED_KEYS = [
    "name", "version", "model_type", "provenance", "trained_on", "feature_order",
    "classes", "scaler_mean", "scaler_scale", "coefficients", "intercepts",
]


class ArtifactError(RuntimeError):
    """The model artifact is missing, unreadable or inconsistent."""


def validate_artifact(artifact: dict) -> None:
    missing = [key for key in REQUIRED_KEYS if key not in artifact]
    if missing:
        raise ArtifactError(f"Model artifact is missing keys: {missing}")

    if artifact["feature_order"] != FEATURE_ORDER:
        raise ArtifactError(
            "The artifact's feature order does not match the current feature pipeline. "
            f"Artifact: {artifact['feature_order']}; pipeline: {FEATURE_ORDER}."
        )

    n_classes, n_features = len(artifact["classes"]), len(artifact["feature_order"])
    coef = np.asarray(artifact["coefficients"], dtype="float64")
    shapes_ok = (
        coef.shape == (n_classes, n_features)
        and len(artifact["intercepts"]) == n_classes
        and len(artifact["scaler_mean"]) == n_features
        and len(artifact["scaler_scale"]) == n_features
    )
    if not shapes_ok:
        raise ArtifactError("Model artifact arrays have inconsistent shapes.")
    if not np.all(np.isfinite(coef)) or np.any(np.asarray(artifact["scaler_scale"]) <= 0):
        raise ArtifactError("Model artifact contains non-finite coefficients or invalid scales.")


def load_artifact(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise ArtifactError(f"Model artifact not found: {path}")
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ArtifactError(f"Model artifact {path.name} is not valid JSON: {error}") from error
    validate_artifact(artifact)
    return artifact


def save_artifact(artifact: dict, path: Path) -> None:
    validate_artifact(artifact)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")


def load_metrics(path: Path) -> dict | None:
    """Evaluation metrics saved next to the artifact, or None if absent/unreadable."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
