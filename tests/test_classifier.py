import copy
import json

import numpy as np
import pytest

from src.config import resolve_path
from src.features.feature_engineering import FEATURE_ORDER, FeatureSet
from src.models.artifacts import ArtifactError, load_artifact, validate_artifact
from src.models.classifier import (
    INCONCLUSIVE, LogisticRegressionClassifier, RuleBasedClassifier, softmax,
)
from src.models.inference import build_classifier, predict_with_uncertainty
from src.pipeline import compute_sample_features


@pytest.fixture(scope="module")
def artifact(settings):
    return load_artifact(resolve_path(settings["model"]["artifact_path"]))


def features_for(make_sample, settings, scenario, **kwargs) -> FeatureSet:
    _, _, features = compute_sample_features(make_sample(scenario, **kwargs), settings)
    return features


def test_bundled_artifact_is_valid_and_synthetic(artifact):
    validate_artifact(artifact)
    assert artifact["feature_order"] == FEATURE_ORDER
    assert artifact["trained_on"]["data"] == "synthetic"
    assert "SYNTHETIC" in artifact["provenance"]


def test_softmax_sums_to_one():
    probs = softmax(np.array([1000.0, 1001.0, 999.0]))  # large values: must not overflow
    assert probs.sum() == pytest.approx(1.0)


def test_logistic_probabilities_valid(artifact, make_sample, settings):
    model = LogisticRegressionClassifier(artifact)
    output = model.predict(features_for(make_sample, settings, "water_stress").values)
    assert sum(output.probabilities.values()) == pytest.approx(1.0)
    assert output.category == max(output.probabilities, key=output.probabilities.get)


def test_linear_contributions_reconstruct_the_score(artifact, make_sample, settings):
    model = LogisticRegressionClassifier(artifact)
    output = model.predict(features_for(make_sample, settings, "pest_stress").values)
    e = output.explanation
    reconstructed = e["intercept"] + sum(e["contributions"].values())
    assert reconstructed == pytest.approx(e["logits"][e["explained_class"]])


def test_numpy_inference_matches_sklearn(artifact, make_sample, settings):
    from sklearn.linear_model import LogisticRegression

    reference = LogisticRegression()
    reference.classes_ = np.array(artifact["classes"])
    reference.coef_ = np.array(artifact["coefficients"])
    reference.intercept_ = np.array(artifact["intercepts"])
    features = features_for(make_sample, settings, "disease_stress")
    z = (features.as_vector() - np.array(artifact["scaler_mean"])) / np.array(artifact["scaler_scale"])
    expected = reference.predict_proba(z.reshape(1, -1))[0]
    ours = LogisticRegressionClassifier(artifact).predict(features.values).probabilities
    assert np.allclose([ours[c] for c in artifact["classes"]], expected)


@pytest.mark.parametrize("scenario", ["healthy", "water_stress", "nutrient_stress", "pest_stress", "disease_stress"])
def test_clear_examples_classified_by_both_classifiers(scenario, artifact, make_sample, settings):
    features = features_for(make_sample, settings, scenario, seed=3)
    ml = predict_with_uncertainty(features, LogisticRegressionClassifier(artifact), settings)
    rules = predict_with_uncertainty(features, RuleBasedClassifier(settings["rule_classifier"]), settings)
    assert ml.category == scenario
    assert rules.category == scenario


def test_predictions_are_reproducible(artifact, make_sample, settings):
    model = LogisticRegressionClassifier(artifact)
    first = model.predict(features_for(make_sample, settings, "nutrient_stress", seed=11).values)
    second = model.predict(features_for(make_sample, settings, "nutrient_stress", seed=11).values)
    assert first.probabilities == second.probabilities


def test_missing_artifact_falls_back_to_rules(settings, tmp_path):
    broken = copy.deepcopy(settings)
    broken["model"]["artifact_path"] = str(tmp_path / "does_not_exist.json")
    classifier, notices = build_classifier("ml_demo", broken)
    assert classifier.mode == "rule_based"
    assert notices and "could not be loaded" in notices[0]


def test_corrupt_artifact_falls_back_to_rules(settings, tmp_path):
    path = tmp_path / "corrupt.json"
    path.write_text("{not json", encoding="utf-8")
    broken = copy.deepcopy(settings)
    broken["model"]["artifact_path"] = str(path)
    classifier, notices = build_classifier("ml_demo", broken)
    assert classifier.mode == "rule_based" and notices


def test_artifact_with_wrong_feature_order_rejected(artifact, tmp_path):
    bad = copy.deepcopy(artifact)
    bad["feature_order"] = list(reversed(bad["feature_order"]))
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ArtifactError, match="feature order"):
        load_artifact(path)


def test_low_valid_fraction_is_inconclusive(artifact, settings):
    features = FeatureSet({name: 0.5 for name in FEATURE_ORDER}, n_valid=10, n_total=100)
    prediction = predict_with_uncertainty(features, LogisticRegressionClassifier(artifact), settings)
    assert prediction.category == INCONCLUSIVE
    assert prediction.raw_category is None  # classifier never ran
    assert "valid" in prediction.reasons[0]


def test_missing_feature_is_inconclusive(artifact, make_sample, settings):
    features = features_for(make_sample, settings, "healthy")
    features.values["ndre_mean"] = float("nan")
    prediction = predict_with_uncertainty(features, LogisticRegressionClassifier(artifact), settings)
    assert prediction.is_inconclusive
    assert any("ndre_mean" in reason for reason in prediction.reasons)


def test_out_of_distribution_input_is_inconclusive(artifact, make_sample, settings):
    features = features_for(make_sample, settings, "healthy")
    features.values["ndvi_std"] = 2.0  # impossible for the training data
    prediction = predict_with_uncertainty(features, LogisticRegressionClassifier(artifact), settings)
    assert prediction.is_inconclusive
    assert any("outside the range" in reason for reason in prediction.reasons)


def test_low_confidence_is_inconclusive(artifact, make_sample, settings):
    strict = copy.deepcopy(settings)
    strict["uncertainty"]["min_confidence"] = 0.999
    features = features_for(make_sample, settings, "disease_stress")
    prediction = predict_with_uncertainty(features, LogisticRegressionClassifier(artifact), strict)
    assert prediction.category == INCONCLUSIVE
    assert prediction.raw_category == "disease_stress"


def test_rule_classifier_no_match_is_inconclusive(settings):
    rules = RuleBasedClassifier(settings["rule_classifier"])
    # Uniform canopy, high NDVI but a borderline chlorophyll ratio matches no rule.
    values = {name: 0.0 for name in FEATURE_ORDER}
    values.update(ndvi_mean=0.8, ndre_ndvi_ratio=0.275, ndvi_std=0.02, low_vigor_fraction=0.0)
    features = FeatureSet(values, 100, 100)
    prediction = predict_with_uncertainty(features, rules, settings)
    assert prediction.category == INCONCLUSIVE
    assert prediction.explanation["fired_rule"] is None


def test_rule_trace_lists_every_rule(settings, make_sample):
    rules = RuleBasedClassifier(settings["rule_classifier"])
    output = rules.predict(features_for(make_sample, settings, "pest_stress").values)
    assert [r["rule_id"] for r in output.explanation["trace"]] == ["R1", "R2", "R3", "R4", "R5"]
    assert output.explanation["fired_rule"] == "R1"


def test_non_crop_surface_is_inconclusive(artifact, settings):
    values = {name: 0.1 for name in FEATURE_ORDER}
    values["ndvi_mean"] = 0.15
    prediction = predict_with_uncertainty(FeatureSet(values, 100, 100), LogisticRegressionClassifier(artifact), settings)
    assert prediction.is_inconclusive
    assert any("non-crop" in reason for reason in prediction.reasons)
