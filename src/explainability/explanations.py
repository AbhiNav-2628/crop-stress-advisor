"""Explanations built only from values the pipeline actually computed.

* Logistic regression: each feature's contribution to the predicted class
  score is coefficient x standardized value. For a linear model this is an
  exact decomposition of the score (not an approximation such as SHAP
  sampling), measured relative to an average synthetic training sample.
* Rule-based classifier: the evaluated conditions of every rule, with the
  actual feature value next to each threshold.

A contribution says which inputs moved the model's score. It does NOT say
what biologically caused the crop's condition.
"""

import numpy as np

from src.features.feature_engineering import FEATURE_DESCRIPTIONS
from src.models.inference import Prediction

GENERAL_LIMITATIONS = [
    "Vegetation indices respond to many causes; similar spectral changes can come from "
    "water deficit, nutrient deficiency, pests, disease, phenology, crop variety or soil background.",
    "The active classifier was fitted to / tuned on SYNTHETIC data and has not been validated "
    "against field observations. Treat the output as a prompt for inspection, not a diagnosis.",
    "Feature attributions explain the model's arithmetic, not the biological cause of stress.",
    "Results depend on correct radiometric calibration (surface reflectance) and correct band mapping.",
]

# Generic, crop-independent readings of NDVI used only to give context.
# Healthy NDVI values depend strongly on crop, growth stage and sensor.
NDVI_READINGS = [
    (0.2, "very low - bare soil, water or almost no green cover"),
    (0.4, "low - sparse or weak green cover"),
    (0.6, "moderate green cover"),
    (1.01, "high - dense green canopy"),
]


def describe_ndvi(value: float) -> str:
    if not np.isfinite(value):
        return "not available"
    for upper, text in NDVI_READINGS:
        if value < upper:
            return text
    return NDVI_READINGS[-1][1]


def _contribution_rows(explanation: dict, features: dict) -> list[dict]:
    rows = []
    for name, contribution in explanation["contributions"].items():
        rows.append({
            "feature": name,
            "description": FEATURE_DESCRIPTIONS.get(name, ""),
            "value": features.get(name),
            "zscore": explanation["zscores"][name],
            "contribution": contribution,
        })
    rows.sort(key=lambda row: abs(row["contribution"]), reverse=True)
    return rows


def _format_condition(c: dict) -> dict:
    value = "n/a" if not np.isfinite(c["value"]) else f"{c['value']:.3f}"
    return {"text": f"{c['feature']} {c['op']} {c['threshold']} (actual {value})", "passed": c["passed"]}


def explain_prediction(prediction: Prediction, features: dict) -> dict:
    """Assemble a display- and report-friendly explanation dictionary."""
    explanation = {
        "method": prediction.explanation.get("type", "none"),
        "headline": "",
        "contributions": [],
        "supporting": [],
        "opposing": [],
        "rule_trace": [],
        "inconclusive_reasons": list(prediction.reasons),
        "uncertainty_notes": list(prediction.uncertainty_notes),
        "ndvi_reading": describe_ndvi(features.get("ndvi_mean", float("nan"))),
        "limitations": list(GENERAL_LIMITATIONS),
    }

    if prediction.raw_category is None:
        explanation["method"] = "none"
        explanation["headline"] = (
            "The classifier was not run because the input failed quality checks."
        )
        return explanation

    if explanation["method"] == "linear_contributions":
        rows = _contribution_rows(prediction.explanation, features)
        explanation["contributions"] = rows
        explanation["supporting"] = [r for r in rows if r["contribution"] > 0][:3]
        explanation["opposing"] = [r for r in rows if r["contribution"] < 0][:3]
        target = prediction.explanation["explained_class"]
        top = ", ".join(r["feature"] for r in explanation["supporting"]) or "none"
        explanation["headline"] = (
            f"The model's score for '{target}' was raised most by: {top} "
            "(relative to an average synthetic training plot)."
        )

    elif explanation["method"] == "rule_trace":
        trace = prediction.explanation["trace"]
        explanation["rule_trace"] = [
            {
                "rule_id": rule["rule_id"],
                "category": rule["category"],
                "description": rule["description"],
                "matched": rule["matched"],
                "all_of": [_format_condition(c) for c in rule["all_of"]],
                "any_of": [_format_condition(c) for c in rule["any_of"]],
            }
            for rule in trace
        ]
        fired = prediction.explanation.get("fired_rule")
        if fired:
            rule = next(r for r in trace if r["rule_id"] == fired)
            explanation["headline"] = (
                f"Rule {fired} fired: {rule['description']}. Rules are checked in order "
                "and the first fully satisfied rule decides."
            )
        else:
            explanation["headline"] = "No rule's conditions were all satisfied."

    if prediction.is_inconclusive and prediction.raw_category:
        explanation["headline"] += (
            f" The classifier's raw answer was '{prediction.raw_category}', but it was "
            "withheld as inconclusive (see reasons)."
        )
    return explanation
