"""Stress classifiers that share one small interface.

Every classifier receives a {feature_name: value} dictionary (see
FEATURE_ORDER) and returns a ClassifierOutput. Uncertainty handling and the
final "inconclusive" decision happen afterwards in `inference.py`, so the
classifiers themselves stay simple.

Two implementations are provided:

* LogisticRegressionClassifier - a multinomial logistic regression whose
  coefficients are loaded from a JSON artifact. Inference is plain NumPy, so
  no pickled scikit-learn object is needed at run time.
* RuleBasedClassifier - a transparent decision list with thresholds from
  config/settings.yaml.
"""

import operator
from dataclasses import dataclass, field

import numpy as np

from src.features.feature_engineering import FEATURE_ORDER

INCONCLUSIVE = "inconclusive"

OPERATORS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


@dataclass
class ClassifierOutput:
    category: str
    probabilities: dict[str, float] | None  # None for classifiers without probabilities
    explanation: dict = field(default_factory=dict)


class StressClassifier:
    """Base class documenting the interface every classifier provides."""

    name: str = "base"
    mode: str = "base"
    version: str = "0"
    provenance: str = ""
    feature_names: list[str] = FEATURE_ORDER
    classes: list[str] = []

    def predict(self, features: dict[str, float]) -> ClassifierOutput:
        raise NotImplementedError

    def info(self) -> dict:
        return {
            "name": self.name,
            "mode": self.mode,
            "version": self.version,
            "provenance": self.provenance,
            "classes": list(self.classes),
            "feature_order": list(self.feature_names),
        }


# --------------------------------------------------------------------------
# Logistic regression (coefficients from a JSON artifact)
# --------------------------------------------------------------------------

def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()  # subtract the max for numerical stability
    exp = np.exp(shifted)
    return exp / exp.sum()


class LogisticRegressionClassifier(StressClassifier):
    mode = "ml_demo"

    def __init__(self, artifact: dict):
        self.name = artifact["name"]
        self.version = artifact["version"]
        self.provenance = artifact["provenance"]
        self.feature_names = artifact["feature_order"]
        self.classes = artifact["classes"]
        self.mean = np.array(artifact["scaler_mean"], dtype="float64")
        self.scale = np.array(artifact["scaler_scale"], dtype="float64")
        self.coef = np.array(artifact["coefficients"], dtype="float64")       # (classes, features)
        self.intercept = np.array(artifact["intercepts"], dtype="float64")    # (classes,)
        self.artifact = artifact

    def standardize(self, features: dict[str, float]) -> np.ndarray:
        """z-scores relative to the (synthetic) training data, in feature order."""
        x = np.array([features[name] for name in self.feature_names], dtype="float64")
        return (x - self.mean) / self.scale

    def predict(self, features: dict[str, float]) -> ClassifierOutput:
        z = self.standardize(features)
        logits = self.coef @ z + self.intercept
        probs = softmax(logits)
        best = int(np.argmax(probs))

        # For a linear model, coef * z is the exact amount each feature adds to
        # the class score, measured against an average training sample (z = 0).
        contributions = {
            name: float(self.coef[best, i] * z[i]) for i, name in enumerate(self.feature_names)
        }
        explanation = {
            "type": "linear_contributions",
            "explained_class": self.classes[best],
            "intercept": float(self.intercept[best]),
            "contributions": contributions,
            "zscores": {name: float(z[i]) for i, name in enumerate(self.feature_names)},
            "logits": {c: float(v) for c, v in zip(self.classes, logits)},
        }
        return ClassifierOutput(
            category=self.classes[best],
            probabilities={c: float(p) for c, p in zip(self.classes, probs)},
            explanation=explanation,
        )


# --------------------------------------------------------------------------
# Transparent rule-based decision list
# --------------------------------------------------------------------------

@dataclass
class Condition:
    feature: str
    op: str
    threshold: float

    def evaluate(self, features: dict[str, float]) -> dict:
        value = features.get(self.feature, float("nan"))
        passed = bool(np.isfinite(value) and OPERATORS[self.op](value, self.threshold))
        return {"feature": self.feature, "op": self.op, "threshold": self.threshold,
                "value": value, "passed": passed}


@dataclass
class DecisionRule:
    rule_id: str
    category: str
    description: str
    all_of: list[Condition] = field(default_factory=list)
    any_of: list[Condition] = field(default_factory=list)  # at least one must hold, if given

    def evaluate(self, features: dict[str, float]) -> dict:
        all_results = [c.evaluate(features) for c in self.all_of]
        any_results = [c.evaluate(features) for c in self.any_of]
        matched = all(r["passed"] for r in all_results) and (
            not any_results or any(r["passed"] for r in any_results)
        )
        return {
            "rule_id": self.rule_id,
            "category": self.category,
            "description": self.description,
            "all_of": all_results,
            "any_of": any_results,
            "matched": matched,
        }


class RuleBasedClassifier(StressClassifier):
    mode = "rule_based"
    name = "Transparent rule-based baseline"
    provenance = (
        "Hand-written decision list. Thresholds were set by inspecting feature "
        "distributions of the SYNTHETIC demonstration data; they are not calibrated "
        "against field observations for any crop."
    )

    def __init__(self, thresholds: dict):
        t = thresholds
        self.version = t.get("version", "rules")
        self.thresholds = t
        self.classes = ["healthy", "water_stress", "nutrient_stress", "pest_stress", "disease_stress"]
        patchy = [
            Condition("ndvi_std", ">=", t["patchy_min_ndvi_std"]),
            Condition("low_vigor_fraction", ">=", t["patchy_min_low_vigor_fraction"]),
        ]
        self.rules = [
            DecisionRule(
                "R1", "pest_stress",
                "Patchy canopy where weak patches look like exposed soil (defoliation-like pattern)",
                all_of=[Condition("ndre_ndvi_ratio", ">=", t["pest_min_ndre_ratio"])],
                any_of=patchy,
            ),
            DecisionRule(
                "R2", "disease_stress",
                "Patchy canopy where weak patches keep a chlorosis-like signature (lesion-like foci)",
                all_of=[Condition("ndre_ndvi_ratio", "<", t["pest_min_ndre_ratio"])],
                any_of=patchy,
            ),
            DecisionRule(
                "R3", "healthy",
                "High, uniform vigour with a normal chlorophyll signal",
                all_of=[
                    Condition("ndvi_mean", ">=", t["healthy_min_ndvi"]),
                    Condition("ndre_ndvi_ratio", ">=", t["healthy_min_ndre_ratio"]),
                ],
            ),
            DecisionRule(
                "R4", "nutrient_stress",
                "Uniform canopy whose chlorophyll signal (NDRE) is low relative to biomass (NDVI)",
                all_of=[Condition("ndre_ndvi_ratio", "<", t["nutrient_max_ndre_ratio"])],
            ),
            DecisionRule(
                "R5", "water_stress",
                "Uniform loss of vigour without a specific chlorophyll signature",
                all_of=[
                    Condition("ndvi_mean", "<", t["water_max_ndvi"]),
                    Condition("ndre_ndvi_ratio", ">=", t["nutrient_max_ndre_ratio"]),
                ],
            ),
        ]

    def predict(self, features: dict[str, float]) -> ClassifierOutput:
        trace = [rule.evaluate(features) for rule in self.rules]
        fired = next((r for r in trace if r["matched"]), None)
        category = fired["category"] if fired else INCONCLUSIVE
        explanation = {
            "type": "rule_trace",
            "fired_rule": fired["rule_id"] if fired else None,
            "trace": trace,
        }
        return ClassifierOutput(category=category, probabilities=None, explanation=explanation)
