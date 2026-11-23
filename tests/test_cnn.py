"""Automated tests for the Deep Learning Convolutional Neural Network (CNN) integration."""

import numpy as np
import pytest
import torch

from src.config import resolve_path
from src.features.feature_engineering import FEATURE_ORDER, FeatureSet
from src.models.classifier import INCONCLUSIVE
from src.models.cnn_classifier import CNNStressClassifier, load_cnn_classifier
from src.models.cnn_model import Multispectral1DCNN, MultispectralSpatial2DCNN
from src.models.inference import build_classifier, predict_with_uncertainty
from src.pipeline import analyze_sample, compute_sample_features


@pytest.fixture(scope="module")
def cnn_classifier(settings):
    cnn_path = resolve_path(settings["cnn_model"]["artifact_path"])
    return load_cnn_classifier(cnn_path)


def features_for(make_sample, settings, scenario, **kwargs) -> FeatureSet:
    _, _, features = compute_sample_features(make_sample(scenario, **kwargs), settings)
    return features


def test_cnn_1d_forward_pass():
    model = Multispectral1DCNN(in_features=9, num_classes=5)
    model.eval()
    x = torch.randn(4, 9)
    out = model(x)
    assert out.shape == (4, 5)

    x_3d = torch.randn(4, 1, 9)
    out_3d = model(x_3d)
    assert out_3d.shape == (4, 5)


def test_cnn_2d_forward_pass():
    model = MultispectralSpatial2DCNN(in_channels=5, num_classes=5)
    model.eval()
    x = torch.randn(2, 5, 32, 32)
    out = model(x)
    assert out.shape == (2, 5)


def test_cnn_gradcam_generation():
    model = MultispectralSpatial2DCNN(in_channels=5, num_classes=5)
    x = torch.randn(1, 5, 32, 32)
    cam, target = model.generate_gradcam(x)
    assert cam.shape == (32, 32)
    assert 0 <= target < 5
    assert np.all(cam >= 0.0)
    assert np.all(cam <= 1.0)


def test_cnn_saliency_computation():
    model = Multispectral1DCNN(in_features=9, num_classes=5)
    x = torch.randn(9)
    saliency, target = model.compute_saliency(x)
    assert saliency.shape == (9,)
    assert 0 <= target < 5
    assert np.all(np.isfinite(saliency))


def test_cnn_classifier_metadata(cnn_classifier):
    info = cnn_classifier.info()
    assert info["mode"] == "cnn_deep_learning"
    assert info["name"] == "Deep Learning Multispectral CNN"
    assert len(info["classes"]) == 5
    assert info["feature_order"] == FEATURE_ORDER


def test_cnn_probabilities_sum_to_one(cnn_classifier, make_sample, settings):
    features = features_for(make_sample, settings, "water_stress")
    output = cnn_classifier.predict(features.values)
    assert sum(output.probabilities.values()) == pytest.approx(1.0, abs=1e-5)
    assert output.category == max(output.probabilities, key=output.probabilities.get)


def test_cnn_predictions_are_reproducible(cnn_classifier, make_sample, settings):
    features = features_for(make_sample, settings, "nutrient_stress", seed=17)
    out1 = cnn_classifier.predict(features.values)
    out2 = cnn_classifier.predict(features.values)
    assert out1.category == out2.category
    for c in cnn_classifier.classes:
        assert out1.probabilities[c] == pytest.approx(out2.probabilities[c], abs=1e-6)


@pytest.mark.parametrize("scenario", ["healthy", "water_stress", "nutrient_stress", "pest_stress", "disease_stress"])
def test_cnn_classifies_clear_synthetic_examples(scenario, cnn_classifier, make_sample, settings):
    features = features_for(make_sample, settings, scenario, seed=3)
    prediction = predict_with_uncertainty(features, cnn_classifier, settings)
    assert prediction.category == scenario


def test_cnn_uncertainty_gating_low_confidence(cnn_classifier, settings):
    # Equal probabilities -> low confidence -> inconclusive
    mock_features = FeatureSet(
        values={name: 0.5 for name in FEATURE_ORDER},
        n_valid=4096,
        n_total=4096,
        warnings=[],
    )
    # With tight limits
    strict_settings = dict(settings)
    strict_settings["uncertainty"] = dict(settings["uncertainty"], min_confidence=0.99)
    pred = predict_with_uncertainty(mock_features, cnn_classifier, strict_settings)
    assert pred.is_inconclusive
    assert pred.category == INCONCLUSIVE


def test_cnn_uncertainty_gating_out_of_distribution(cnn_classifier, settings):
    extreme_features = FeatureSet(
        values={name: 100.0 for name in FEATURE_ORDER},
        n_valid=4096,
        n_total=4096,
        warnings=[],
    )
    pred = predict_with_uncertainty(extreme_features, cnn_classifier, settings)
    assert pred.is_inconclusive
    assert any("outside the range" in r for r in pred.reasons)


def test_cnn_spatial_raster_prediction(cnn_classifier, make_sample):
    sample = make_sample("pest_stress", seed=5)
    cube = np.stack([sample.bands[b] for b in ["blue", "green", "red", "red_edge", "nir"]])
    output, heatmap = cnn_classifier.predict_spatial_raster(cube)
    assert output.category in cnn_classifier.classes
    assert heatmap is not None
    assert heatmap.shape == sample.shape


def test_cnn_in_full_pipeline(cnn_classifier, make_sample, settings, advisory_rules):
    sample = make_sample("nutrient_stress")
    result = analyze_sample(sample, cnn_classifier, settings, advisory_rules)
    assert result.prediction.category == "nutrient_stress"
    assert result.explanation["method"] == "cnn_explanation"
    assert len(result.explanation["contributions"]) == 9
    assert len(result.advisories) >= 1
    assert result.advisories[-1].rule_id == "ADV-GEN-01"


def test_build_classifier_returns_cnn(settings):
    classifier, notices = build_classifier("cnn_deep_learning", settings)
    assert not notices
    assert isinstance(classifier, CNNStressClassifier)
    assert classifier.mode == "cnn_deep_learning"
