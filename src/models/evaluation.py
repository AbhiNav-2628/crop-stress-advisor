"""Evaluation metrics for classifiers on labelled samples.

These functions only compute metrics; they never invent them. When they are
run on synthetic data, the results describe performance on synthetic data
only and say nothing about real fields.
"""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from src.models.classifier import INCONCLUSIVE


def classification_metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    """Accuracy, per-class precision/recall/F1 and a confusion matrix.

    Predictions of "inconclusive" count as errors in the overall figures and
    are also reported separately as coverage (the share of samples the system
    was willing to classify).
    """
    y_true = list(y_true)
    y_pred = list(y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    matrix_labels = labels + ([INCONCLUSIVE] if INCONCLUSIVE in y_pred else [])
    matrix = confusion_matrix(y_true, y_pred, labels=matrix_labels)

    decided = [i for i, p in enumerate(y_pred) if p != INCONCLUSIVE]
    accuracy_on_decided = (
        accuracy_score([y_true[i] for i in decided], [y_pred[i] for i in decided]) if decided else None
    )
    return {
        "n_samples": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(np.mean(f1)),
        "coverage": len(decided) / len(y_true) if y_true else 0.0,
        "accuracy_on_decided": accuracy_on_decided,
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        },
        "confusion_matrix": {"labels": matrix_labels, "matrix": matrix.tolist()},
    }
