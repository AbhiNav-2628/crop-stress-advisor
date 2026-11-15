"""End-to-end checks on the bundled demonstration files, using the same
loading code as the command-line demo."""

import pytest

from scripts.run_demo_cli import load_demo_samples
from src.models.inference import build_classifier
from src.pipeline import analyze_sample


@pytest.fixture(scope="module")
def results_by_mode(settings, advisory_rules):
    samples = load_demo_samples(settings)
    out = {}
    for mode in ("ml_demo", "rule_based"):
        classifier, notices = build_classifier(mode, settings)
        assert not notices, notices
        out[mode] = {
            (s.source_name, s.sample_id): analyze_sample(s, classifier, settings, advisory_rules)
            for s in samples
        }
    return out


EXPECTED = {
    ("plot_healthy.tif", "plot_healthy"): "healthy",
    ("plot_water_stress.tif", "plot_water_stress"): "water_stress",
    ("plot_nutrient_stress.tif", "plot_nutrient_stress"): "nutrient_stress",
    ("plot_nutrient_stress_uint16_x10000.tif", "plot_nutrient_stress_uint16_x10000"): "nutrient_stress",
    ("plot_pest_stress.tif", "plot_pest_stress"): "pest_stress",
    ("plot_disease_stress.tif", "plot_disease_stress"): "disease_stress",
    ("plot_cloud_masked.tif", "plot_cloud_masked"): "inconclusive",
    ("plot_bare_soil.tif", "plot_bare_soil"): "inconclusive",
    ("demo_points.csv", "P01-healthy"): "healthy",
    ("demo_points.csv", "P02-water_stress"): "water_stress",
    ("demo_points.csv", "P03-nutrient_stress"): "nutrient_stress",
    ("demo_points.csv", "P04-pest_stress"): "pest_stress",
    ("demo_points.csv", "P05-disease_stress"): "disease_stress",
    ("demo_points_no_red_edge.csv", "P01-healthy"): "inconclusive",
}


@pytest.mark.parametrize("mode", ["ml_demo", "rule_based"])
@pytest.mark.parametrize("key,expected", list(EXPECTED.items()))
def test_demo_file_outcomes(results_by_mode, mode, key, expected):
    assert results_by_mode[mode][key].prediction.category == expected


def test_separate_band_files_match_multiband_file(results_by_mode):
    results = results_by_mode["ml_demo"]
    band_files = next(r for (src, sid), r in results.items() if sid == "band-files")
    multiband = results[("plot_water_stress.tif", "plot_water_stress")]
    # Same plot, written two ways (generated from the same cube) -> same features.
    assert band_files.prediction.category == "water_stress"
    assert band_files.features.values["ndvi_mean"] == pytest.approx(
        multiband.features.values["ndvi_mean"], abs=0.05)


def test_every_result_has_general_advisory(results_by_mode):
    for result in results_by_mode["ml_demo"].values():
        assert result.advisories[-1].rule_id == "ADV-GEN-01"


def test_missing_red_edge_still_computes_other_indices(results_by_mode):
    result = results_by_mode["ml_demo"][("demo_points_no_red_edge.csv", "P02-water_stress")]
    assert {"ndvi", "gndvi", "savi", "evi"} == set(result.indices)
    assert "ndre" in result.skipped_indices
