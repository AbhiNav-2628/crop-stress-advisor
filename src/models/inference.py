"""Choosing a classifier and turning its raw output into a final, gated prediction.

The final category is "inconclusive" whenever:
  * the input quality is too poor (too few valid pixels, non-crop surface),
  * a feature the classifier needs could not be calculated,
  * the input lies outside the range the model has seen (out-of-distribution),
  * the model's probability or margin is too low, or
  * no rule of the rule-based classifier matched.
"""

from dataclasses import dataclass, field

import numpy as np

from src.config import resolve_path
from src.features.feature_engineering import FeatureSet
from src.models.artifacts import ArtifactError, load_artifact
from src.models.classifier import (
    INCONCLUSIVE,
    LogisticRegressionClassifier,
    RuleBasedClassifier,
    StressClassifier,
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

MODE_LABELS = {
    "ml_demo": "Demonstration ML model (logistic regression fitted to synthetic data)",
    "rule_based": "Transparent rule-based baseline",
}


@dataclass
class Prediction:
    category: str
    raw_category: str | None          # what the classifier said before gating
    confidence: float | None          # top-class probability (None for rules)
    probabilities: dict[str, float] | None
    is_inconclusive: bool
    reasons: list[str] = field(default_factory=list)       # why it is inconclusive
    uncertainty_notes: list[str] = field(default_factory=list)
    explanation: dict = field(default_factory=dict)
    classifier: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "raw_category": self.raw_category,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "is_inconclusive": self.is_inconclusive,
            "inconclusive_reasons": self.reasons,
            "uncertainty_notes": self.uncertainty_notes,
            "classifier": self.classifier,
        }


def build_classifier(mode: str, settings: dict) -> tuple[StressClassifier, list[str]]:
    """Return (classifier, notices). Falls back to rules if the ML artifact is unusable."""
    notices = []
    if mode == "ml_demo":
        try:
            artifact = load_artifact(resolve_path(settings["model"]["artifact_path"]))
            return LogisticRegressionClassifier(artifact), notices
        except ArtifactError as error:
            logger.warning("Falling back to the rule-based classifier: %s", error)
            notices.append(
                f"The demonstration ML model could not be loaded ({error}). "
                "The transparent rule-based baseline is being used instead."
            )
    elif mode != "rule_based":
        notices.append(f"Unknown classifier mode '{mode}'; using the rule-based baseline.")
    return RuleBasedClassifier(settings["rule_classifier"]), notices


def quality_blockers(features: FeatureSet, classifier: StressClassifier, settings: dict) -> list[str]:
    """Reasons that stop classification before the classifier is called."""
    reasons = []
    quality = settings["quality"]
    if features.valid_fraction < quality["min_valid_fraction"]:
        reasons.append(
            f"Only {features.valid_fraction:.0%} of observations are valid; at least "
            f"{quality['min_valid_fraction']:.0%} is required."
        )
    missing = [name for name in classifier.feature_names if name in features.missing]
    if missing:
        reasons.append(
            "Required feature(s) could not be calculated: " + ", ".join(missing)
            + ". This usually means a band (e.g. Red Edge or Blue) was not provided."
        )
    ndvi_mean = features.values.get("ndvi_mean", float("nan"))
    if np.isfinite(ndvi_mean) and ndvi_mean < quality["min_canopy_ndvi"]:
        reasons.append(
            f"Mean NDVI is {ndvi_mean:.2f}, below {quality['min_canopy_ndvi']}; the area is "
            "probably bare soil, water or another non-crop surface, which is unsupported."
        )
    return reasons


def predict_with_uncertainty(
    features: FeatureSet, classifier: StressClassifier, settings: dict
) -> Prediction:
    info = classifier.info()
    blockers = quality_blockers(features, classifier, settings)
    if blockers:
        return Prediction(INCONCLUSIVE, None, None, None, True, blockers, classifier=info)

    output = classifier.predict(features.values)
    reasons, notes = [], []
    confidence = None
    limits = settings["uncertainty"]

    if output.probabilities is not None:
        ranked = sorted(output.probabilities.items(), key=lambda item: item[1], reverse=True)
        confidence = ranked[0][1]
        margin = ranked[0][1] - ranked[1][1]
        notes.append(
            f"Top probability {ranked[0][1]:.2f} ({ranked[0][0]}); runner-up "
            f"{ranked[1][1]:.2f} ({ranked[1][0]}); margin {margin:.2f}."
        )
        if confidence < limits["min_confidence"]:
            reasons.append(
                f"Top probability {confidence:.2f} is below the confidence threshold "
                f"{limits['min_confidence']}."
            )
        if margin < limits["min_margin"]:
            reasons.append(
                f"'{ranked[0][0]}' and '{ranked[1][0]}' are too close to separate "
                f"(margin {margin:.2f} < {limits['min_margin']})."
            )
        zscores = output.explanation.get("zscores", {})
        extreme = {k: v for k, v in zscores.items() if abs(v) > limits["max_abs_zscore"]}
        if extreme:
            described = ", ".join(f"{k} (z={v:+.1f})" for k, v in extreme.items())
            reasons.append(
                "Input is outside the range of the model's synthetic training data: "
                f"{described}. The model cannot be trusted here."
            )
    else:
        notes.append("The rule-based classifier gives no probability; see the rule trace.")
        if output.category == INCONCLUSIVE:
            reasons.append("No decision rule matched this combination of feature values.")

    for warning in features.warnings:
        notes.append(f"Data-quality note: {warning}")

    final = INCONCLUSIVE if reasons else output.category
    return Prediction(
        category=final,
        raw_category=output.category,
        confidence=confidence,
        probabilities=output.probabilities,
        is_inconclusive=bool(reasons),
        reasons=reasons,
        uncertainty_notes=notes,
        explanation=output.explanation,
        classifier=info,
    )
