"""Train the Deep Learning Multispectral CNN model (1D feature CNN + 2D spatial CNN).

Saves the trained PyTorch checkpoint to models/demo_cnn_v1.pt and the
test set evaluation metrics to models/demo_cnn_v1_metrics.json.

Run from project root:
    python scripts/train_cnn_model.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_settings, resolve_path  # noqa: E402
from src.data.ingestion import RasterInput  # noqa: E402
from src.data.preprocessing import sample_from_raster  # noqa: E402
from src.data.synthetic import SCENARIOS, generate_dataset  # noqa: E402
from src.features.feature_engineering import FEATURE_ORDER  # noqa: E402
from src.models.cnn_classifier import CNNStressClassifier  # noqa: E402
from src.models.cnn_model import Multispectral1DCNN, MultispectralSpatial2DCNN  # noqa: E402
from src.models.evaluation import classification_metrics  # noqa: E402
from src.models.inference import predict_with_uncertainty  # noqa: E402
from src.pipeline import compute_sample_features  # noqa: E402

TRAINING_SEED = 7
PLOTS_PER_CLASS = 400
TEST_FRACTION = 0.25
DEMO_BAND_MAPPING = {"blue": 1, "green": 2, "red": 3, "red_edge": 4, "nir": 5}


def build_training_data(settings: dict, n_per_class: int, seed: int):
    """Generate synthetic rasters and extract features for training."""
    feature_sets, cubes, labels = [], [], []
    for i, (scenario, cube) in enumerate(generate_dataset(n_per_class, seed)):
        raster = RasterInput(cube, None, f"synthetic-{i}", "synthetic")
        sample = sample_from_raster(raster, DEMO_BAND_MAPPING, settings)
        _, _, features = compute_sample_features(sample, settings)
        feature_sets.append(features)
        cubes.append(cube)
        labels.append(scenario)
    return feature_sets, np.array(cubes, dtype="float32"), labels


def train_1d_cnn(
    X_train: np.ndarray, y_train: np.ndarray, num_classes: int, epochs: int = 40, batch_size: int = 64
) -> Multispectral1DCNN:
    """Train 1D feature CNN on biophysical feature vectors."""
    torch.manual_seed(TRAINING_SEED)
    model = Multispectral1DCNN(in_features=X_train.shape[1], num_classes=num_classes)
    device = torch.device("cpu")
    model.to(device)

    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for epoch in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    return model


def train_2d_cnn(
    cubes_train: np.ndarray, y_train: np.ndarray, num_classes: int, epochs: int = 25, batch_size: int = 32
) -> MultispectralSpatial2DCNN:
    """Train 2D spatial CNN on 5-band multispectral rasters."""
    torch.manual_seed(TRAINING_SEED)
    model = MultispectralSpatial2DCNN(in_channels=5, num_classes=num_classes)
    device = torch.device("cpu")
    model.to(device)

    clean_cubes = np.nan_to_num(cubes_train, nan=0.0, posinf=1.0, neginf=0.0)
    dataset = TensorDataset(
        torch.tensor(clean_cubes, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for epoch in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    return model


def main() -> None:
    settings = load_settings()
    print(f"Generating {PLOTS_PER_CLASS} synthetic plots per class (seed {TRAINING_SEED})...")
    feature_sets, cubes, labels = build_training_data(settings, PLOTS_PER_CLASS, TRAINING_SEED)

    X = np.array([f.as_vector() for f in feature_sets], dtype="float32")
    class_to_idx = {name: i for i, name in enumerate(SCENARIOS)}
    y_indices = np.array([class_to_idx[name] for name in labels], dtype="int64")

    indices = np.arange(len(y_indices))
    train_idx, test_idx = train_test_split(
        indices, test_size=TEST_FRACTION, stratify=y_indices, random_state=TRAINING_SEED
    )

    print(f"Training set: {len(train_idx)} plots | Test set: {len(test_idx)} plots")

    # Fit scaler on training set
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X[train_idx])
    X_test_scaled = scaler.transform(X[test_idx])

    # 1. Train 1D Feature CNN
    print("Training 1D Feature CNN...")
    model_1d = train_1d_cnn(X_train_scaled, y_indices[train_idx], len(SCENARIOS), epochs=45)

    # 2. Train 2D Spatial CNN
    print("Training 2D Spatial Raster CNN...")
    model_2d = train_2d_cnn(cubes[train_idx], y_indices[train_idx], len(SCENARIOS), epochs=25)

    training_metadata = {
        "data": "synthetic",
        "generator": "src/data/synthetic.py",
        "seed": TRAINING_SEED,
        "plots_per_class": PLOTS_PER_CLASS,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "epochs_1d": 45,
        "epochs_2d": 25,
        "optimizer": "AdamW",
        "framework": f"PyTorch {torch.__version__}",
    }

    classifier = CNNStressClassifier(
        model_1d=model_1d,
        model_2d=model_2d,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        classes=SCENARIOS,
        feature_names=FEATURE_ORDER,
        training_metadata=training_metadata,
    )

    # Evaluate on held-out test set
    test_feature_sets = [feature_sets[i] for i in test_idx]
    y_test_true = [labels[i] for i in test_idx]

    # Raw predictions
    y_pred_raw = [classifier.predict(f.values).category for f in test_feature_sets]
    raw_metrics = classification_metrics(y_test_true, y_pred_raw, SCENARIOS)

    # Uncertainty-gated predictions
    y_pred_gated = [
        predict_with_uncertainty(f, classifier, settings).category for f in test_feature_sets
    ]
    gated_metrics = classification_metrics(y_test_true, y_pred_gated, SCENARIOS)

    print(f"\nCNN Evaluation on held-out test set (n={len(test_idx)}):")
    print(f"  Raw Accuracy: {raw_metrics['accuracy']:.3f} | Macro F1: {raw_metrics['macro_f1']:.3f}")
    print(
        f"  Gated Accuracy: {gated_metrics['accuracy']:.3f} | Coverage: {gated_metrics['coverage']:.3f} | "
        f"Accuracy on Decided: {gated_metrics['accuracy_on_decided']:.3f}"
    )

    # Save artifact
    output_pt = resolve_path("models/demo_cnn_v1.pt")
    classifier.save_checkpoint(output_pt)
    print(f"\nSaved CNN model checkpoint to: {output_pt}")

    # Save metrics JSON
    metrics_data = {
        "evaluation_data": "Held-out SYNTHETIC plots (same generator as training data)",
        "warning": (
            "These metrics measure how well the CNN classifier recovers the synthetic scenarios "
            "it was built around. They are NOT evidence of real-world accuracy and must not be "
            "reported as field validation."
        ),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_test": len(test_idx),
        "cnn_raw": raw_metrics,
        "cnn_with_uncertainty_gating": gated_metrics,
    }
    metrics_path = resolve_path("models/demo_cnn_v1_metrics.json")
    metrics_path.write_text(json.dumps(metrics_data, indent=2), encoding="utf-8")
    print(f"Saved CNN evaluation metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
