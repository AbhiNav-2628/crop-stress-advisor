"""The end-to-end analysis for one sample:

SpectralSample -> vegetation indices -> features -> classification with
uncertainty -> explanation -> advisory rules.

The Streamlit app, the command-line demo and the tests all call
`analyze_sample`, so they exercise exactly the same code path.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.advisory.rule_engine import Advisory, build_facts, evaluate_rules
from src.data.preprocessing import SpectralSample
from src.explainability.explanations import explain_prediction
from src.features.feature_engineering import FeatureSet, extract_features, summarize_indices
from src.features.vegetation_indices import compute_indices
from src.models.classifier import StressClassifier
from src.models.inference import Prediction, predict_with_uncertainty


@dataclass
class AnalysisResult:
    sample: SpectralSample
    crop: str
    indices: dict[str, np.ndarray]
    skipped_indices: dict[str, str]
    index_summary: pd.DataFrame
    features: FeatureSet
    prediction: Prediction
    explanation: dict
    advisories: list[Advisory]
    warnings: list[str] = field(default_factory=list)
    analysed_at: str = ""

    @property
    def sample_id(self) -> str:
        return self.sample.sample_id


def compute_sample_features(sample: SpectralSample, settings: dict):
    """Indices and features for a sample. Shared by the app and the training script."""
    index_cfg = settings["indices"]
    indices, skipped = compute_indices(
        sample.bands, sample.valid_mask, index_cfg["savi_l"], index_cfg["min_denominator"]
    )
    features = extract_features(
        indices, sample.valid_mask, index_cfg["low_vigor_ndvi"],
        settings["quality"]["min_observations_for_spatial_stats"],
    )
    return indices, skipped, features


def analyze_sample(
    sample: SpectralSample,
    classifier: StressClassifier,
    settings: dict,
    advisory_rules: list[dict],
    crop: str = "generic",
) -> AnalysisResult:
    indices, skipped, features = compute_sample_features(sample, settings)

    warnings = list(sample.warnings) + list(features.warnings)
    warnings += [f"{name.upper()} not calculated: {reason}." for name, reason in skipped.items()]

    prediction = predict_with_uncertainty(features, classifier, settings)
    explanation = explain_prediction(prediction, features.values)
    facts = build_facts(prediction, features.values, crop, features.valid_fraction, len(warnings))
    advisories = evaluate_rules(advisory_rules, facts)

    return AnalysisResult(
        sample=sample,
        crop=crop,
        indices=indices,
        skipped_indices=skipped,
        index_summary=summarize_indices(indices),
        features=features,
        prediction=prediction,
        explanation=explanation,
        advisories=advisories,
        warnings=warnings,
        analysed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
