"""Train ML and Deep Learning CNN models on a custom multispectral dataset CSV.

Usage:
    python scripts/train_custom_dataset.py --data data/real_world_multispectral_dataset.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_settings
from src.features.feature_engineering import FEATURE_ORDER
from src.models.artifacts import save_artifact

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from src.models.cnn_model import Multispectral1DCNN
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def extract_features_from_df(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Compute standard 9 spectral and biophysical features from bands dataframe."""
    b = df["blue"].values
    g = df["green"].values
    r = df["red"].values
    re = df["red_edge"].values
    nir = df["nir"].values

    # Avoid div-by-zero
    eps = 1e-6
    ndvi = (nir - r) / (nir + r + eps)
    ndre = (nir - re) / (nir + re + eps)
    gndvi = (nir - g) / (nir + g + eps)
    savi = ((nir - r) / (nir + r + 0.5 + eps)) * 1.5
    evi = 2.5 * (nir - r) / (nir + 6.0 * r - 7.5 * b + 1.0 + eps)
    ndre_ndvi_ratio = np.where(np.abs(ndvi) > eps, ndre / ndvi, 0.0)

    # Spatial features for tabular points (approximated by local sample or mean)
    ndvi_std = np.full_like(ndvi, 0.015)
    ndvi_p10 = ndvi - 0.02
    low_vigor = np.where(ndvi < 0.5, 1.0, 0.0)

    features_map = {
        "ndvi_mean": ndvi,
        "ndre_mean": ndre,
        "gndvi_mean": gndvi,
        "savi_mean": savi,
        "evi_mean": evi,
        "ndvi_std": ndvi_std,
        "ndvi_p10": ndvi_p10,
        "low_vigor_fraction": low_vigor,
        "ndre_ndvi_ratio": ndre_ndvi_ratio,
    }

    X = np.column_stack([features_map[k] for k in FEATURE_ORDER]).astype(np.float64)
    y = np.array(df["label"].tolist(), dtype=str)
    return X, y, FEATURE_ORDER


def main() -> None:
    parser = argparse.ArgumentParser(description="Train models on custom multispectral dataset CSV")
    parser.add_argument("--data", default="data/real_world_multispectral_dataset.csv", help="Path to input dataset CSV")
    parser.add_argument("--save-models", action="store_true", default=False, help="Overwrite default production model weights")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Dataset {data_path} not found.")
        sys.exit(1)

    print(f"Loading multispectral dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records across classes: {df['label'].value_counts().to_dict()}")

    X, y, feature_order = extract_features_from_df(df)
    classes = sorted(list(set(y)))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 1. Train Logistic Regression
    print("\n--- Training Machine Learning Model (Logistic Regression) ---")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_s, y_train)
    y_pred_lr = lr.predict(X_test_s)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    f1_lr = f1_score(y_test, y_pred_lr, average="macro")
    print(f"Logistic Regression Accuracy: {acc_lr:.1%} | Macro F1: {f1_lr:.3f}")
    print(classification_report(y_test, y_pred_lr, digits=3))

    # 2. Train Deep Learning CNN if PyTorch available
    if TORCH_AVAILABLE:
        print("\n--- Training Deep Learning CNN (Multispectral1DCNN) ---")
        class_to_idx = {c: i for i, c in enumerate(classes)}
        y_train_idx = np.array([class_to_idx[c] for c in y_train])
        y_test_idx = np.array([class_to_idx[c] for c in y_test])

        t_X_train = torch.tensor(X_train_s, dtype=torch.float32)
        t_y_train = torch.tensor(y_train_idx, dtype=torch.long)
        t_X_test = torch.tensor(X_test_s, dtype=torch.float32)

        model = Multispectral1DCNN(in_features=len(feature_order), num_classes=len(classes))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        model.train()
        for epoch in range(40):
            optimizer.zero_grad()
            out = model(t_X_train)
            loss = criterion(out, t_y_train)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(t_X_test)
            preds_idx = torch.argmax(logits, dim=1).numpy()
            y_pred_cnn = [classes[i] for i in preds_idx]

        acc_cnn = accuracy_score(y_test, y_pred_cnn)
        f1_cnn = f1_score(y_test, y_pred_cnn, average="macro")
        print(f"Deep Learning CNN Accuracy: {acc_cnn:.1%} | Macro F1: {f1_cnn:.3f}")
        print(classification_report(y_test, y_pred_cnn, digits=3))

    print("\nTraining and validation complete!")


if __name__ == "__main__":
    main()
