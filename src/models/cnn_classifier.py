"""Deep Learning Convolutional Neural Network (CNN) Stress Classifier.

Implements the StressClassifier interface using PyTorch 1D and 2D CNN models.
Provides:
  - Standardized biophysical feature classification via Multispectral1DCNN.
  - Direct spatial-spectral raster diagnosis via MultispectralSpatial2DCNN.
  - Saliency attribution (gradient-weighted feature attributions) for explainability.
  - Grad-CAM spatial activation heatmaps when 2D canopy rasters are provided.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.features.feature_engineering import FEATURE_ORDER
from src.models.classifier import ClassifierOutput, StressClassifier
from src.models.cnn_model import (
    DEFAULT_FEATURE_DIM,
    NUM_BANDS,
    NUM_CLASSES,
    Multispectral1DCNN,
    MultispectralSpatial2DCNN,
)


class CNNStressClassifier(StressClassifier):
    """Deep Learning CNN classifier implementing the unified StressClassifier interface."""

    mode: str = "cnn_deep_learning"
    name: str = "Deep Learning Multispectral CNN"
    version: str = "cnn-1.0"
    provenance: str = (
        "Fitted by scripts/train_cnn_model.py using PyTorch on multispectral crop-stress data. "
        "Captures non-linear spectral interactions and spatial lesion/canopy heterogeneity."
    )
    feature_names: list[str] = FEATURE_ORDER
    classes: list[str] = ["healthy", "water_stress", "nutrient_stress", "pest_stress", "disease_stress"]

    def __init__(
        self,
        model_1d: Multispectral1DCNN | None = None,
        model_2d: MultispectralSpatial2DCNN | None = None,
        scaler_mean: np.ndarray | None = None,
        scaler_scale: np.ndarray | None = None,
        classes: list[str] | None = None,
        feature_names: list[str] | None = None,
        training_metadata: dict[str, Any] | None = None,
    ):
        self.device = torch.device("cpu")  # Portable across any CPU or GPU environment
        self.classes = list(classes) if classes is not None else ["healthy", "water_stress", "nutrient_stress", "pest_stress", "disease_stress"]
        self.feature_names = list(feature_names) if feature_names is not None else list(FEATURE_ORDER)

        if scaler_mean is not None:
            self.mean = np.array(scaler_mean, dtype="float32")
        else:
            self.mean = np.zeros(len(self.feature_names), dtype="float32")

        if scaler_scale is not None:
            self.scale = np.array(scaler_scale, dtype="float32")
        else:
            self.scale = np.ones(len(self.feature_names), dtype="float32")

        self.model_1d = model_1d if model_1d is not None else Multispectral1DCNN(
            in_features=len(self.feature_names), num_classes=len(self.classes)
        )
        self.model_1d.to(self.device)
        self.model_1d.eval()

        self.model_2d = model_2d
        if self.model_2d is not None:
            self.model_2d.to(self.device)
            self.model_2d.eval()

        self.training_metadata = training_metadata or {}

    def standardize(self, features: dict[str, float]) -> np.ndarray:
        """z-scores relative to training distribution in feature order."""
        x = np.array([features[name] for name in self.feature_names], dtype="float32")
        # Guard against zero or near-zero scale
        scale = np.where(self.scale > 1e-8, self.scale, 1.0)
        return (x - self.mean) / scale

    def predict(self, features: dict[str, float]) -> ClassifierOutput:
        """Evaluate sample features with the 1D Deep CNN and produce saliency attribution."""
        z = self.standardize(features)
        z_tensor = torch.tensor(z, dtype=torch.float32, device=self.device)

        # Compute output logits and probabilities
        with torch.no_grad():
            self.model_1d.eval()
            logits_tensor = self.model_1d(z_tensor.unsqueeze(0))
            probs_tensor = F.softmax(logits_tensor, dim=1)
            logits = logits_tensor.squeeze(0).cpu().numpy()
            probs = probs_tensor.squeeze(0).cpu().numpy()

        best = int(np.argmax(probs))

        # Compute input-gradient feature attribution (saliency = d(logit)/dz * z)
        try:
            saliency, _ = self.model_1d.compute_saliency(z_tensor, target_class=best)
        except Exception:
            saliency = np.zeros_like(z)

        # Format contributions dictionary for UI and reporting
        contributions = {
            name: float(saliency[i]) for i, name in enumerate(self.feature_names)
        }
        zscores = {
            name: float(z[i]) for i, name in enumerate(self.feature_names)
        }

        explanation = {
            "type": "cnn_explanation",
            "explained_class": self.classes[best],
            "probabilities": {c: float(p) for c, p in zip(self.classes, probs)},
            "logits": {c: float(v) for c, v in zip(self.classes, logits)},
            "zscores": zscores,
            "contributions": contributions,
            "saliency": contributions,
            "model_architecture": "Multispectral 1D-CNN (Conv1D + BatchNorm + LeakyReLU + AdaptivePooling + Dense)",
            "device": str(self.device),
        }

        return ClassifierOutput(
            category=self.classes[best],
            probabilities={c: float(p) for c, p in zip(self.classes, probs)},
            explanation=explanation,
        )

    def predict_spatial_raster(
        self, cube: np.ndarray, band_mapping: dict[str, int] | None = None
    ) -> tuple[ClassifierOutput, np.ndarray | None]:
        """Diagnose 2D spatial raster with MultispectralSpatial2DCNN and generate Grad-CAM."""
        if self.model_2d is None:
            # Fall back to 1D features
            return self.predict({}), None

        # Ensure cube is (5, H, W) in standard band order
        if cube.ndim != 3 or cube.shape[0] < 5:
            raise ValueError(f"Expected multispectral cube with at least 5 bands, got shape {cube.shape}")

        cube_input = cube[:5].astype("float32")
        # Replace non-finite values with 0
        cube_input = np.nan_to_num(cube_input, nan=0.0, posinf=1.0, neginf=0.0)

        tensor_x = torch.tensor(cube_input, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Forward pass
        with torch.no_grad():
            self.model_2d.eval()
            logits = self.model_2d(tensor_x).squeeze(0).cpu().numpy()
            probs = F.softmax(torch.tensor(logits), dim=0).numpy()

        best = int(np.argmax(probs))

        # Grad-CAM heatmap
        try:
            cam_heatmap, _ = self.model_2d.generate_gradcam(tensor_x, target_class=best)
        except Exception:
            cam_heatmap = None

        explanation = {
            "type": "cnn_spatial_gradcam",
            "explained_class": self.classes[best],
            "probabilities": {c: float(p) for c, p in zip(self.classes, probs)},
            "logits": {c: float(v) for c, v in zip(self.classes, logits)},
            "spatial_activation": cam_heatmap.tolist() if cam_heatmap is not None else None,
            "model_architecture": "Multispectral Spatial 2D-CNN (Conv2D + BatchNorm + ReLU + Grad-CAM)",
        }

        output = ClassifierOutput(
            category=self.classes[best],
            probabilities={c: float(p) for c, p in zip(self.classes, probs)},
            explanation=explanation,
        )
        return output, cam_heatmap

    def save_checkpoint(self, path: Path | str) -> None:
        """Save CNN classifier checkpoint to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "version": self.version,
            "name": self.name,
            "provenance": self.provenance,
            "classes": self.classes,
            "feature_names": self.feature_names,
            "scaler_mean": self.mean.tolist(),
            "scaler_scale": self.scale.tolist(),
            "state_dict_1d": self.model_1d.state_dict(),
            "state_dict_2d": self.model_2d.state_dict() if self.model_2d is not None else None,
            "training_metadata": self.training_metadata,
        }
        torch.save(checkpoint, path)

    @classmethod
    def load_from_checkpoint(cls, path: Path | str) -> CNNStressClassifier:
        """Load CNN classifier from a PyTorch checkpoint file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"CNN model checkpoint not found at {path}")

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        classes = checkpoint.get("classes", ["healthy", "water_stress", "nutrient_stress", "pest_stress", "disease_stress"])
        feature_names = checkpoint.get("feature_names", FEATURE_ORDER)

        model_1d = Multispectral1DCNN(in_features=len(feature_names), num_classes=len(classes))
        if "state_dict_1d" in checkpoint and checkpoint["state_dict_1d"] is not None:
            model_1d.load_state_dict(checkpoint["state_dict_1d"])
        model_1d.eval()

        model_2d = None
        if checkpoint.get("state_dict_2d") is not None:
            model_2d = MultispectralSpatial2DCNN(in_channels=NUM_BANDS, num_classes=len(classes))
            model_2d.load_state_dict(checkpoint["state_dict_2d"])
            model_2d.eval()

        scaler_mean = np.array(checkpoint["scaler_mean"], dtype="float32")
        scaler_scale = np.array(checkpoint["scaler_scale"], dtype="float32")

        classifier = cls(
            model_1d=model_1d,
            model_2d=model_2d,
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
            classes=classes,
            feature_names=feature_names,
            training_metadata=checkpoint.get("training_metadata", {}),
        )
        classifier.version = checkpoint.get("version", cls.version)
        classifier.name = checkpoint.get("name", cls.name)
        classifier.provenance = checkpoint.get("provenance", cls.provenance)
        return classifier


def load_cnn_classifier(path: Path | str) -> CNNStressClassifier:
    return CNNStressClassifier.load_from_checkpoint(path)
