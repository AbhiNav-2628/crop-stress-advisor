import numpy as np
import pytest

from src.features.feature_engineering import FEATURE_ORDER, extract_features
from src.features.vegetation_indices import compute_indices, evi, gndvi, ndre, ndvi, safe_divide, savi


def test_ndvi_known_value():
    # (0.5 - 0.1) / (0.5 + 0.1) = 0.6667
    assert ndvi(np.array([0.5]), np.array([0.1]))[0] == pytest.approx(0.4 / 0.6)


def test_ndre_and_gndvi_known_values():
    assert ndre(np.array([0.5]), np.array([0.3]))[0] == pytest.approx(0.2 / 0.8)
    assert gndvi(np.array([0.5]), np.array([0.1]))[0] == pytest.approx(0.4 / 0.6)


def test_savi_formula_with_l_half():
    nir, red = 0.5, 0.1
    expected = (nir - red) / (nir + red + 0.5) * 1.5
    assert savi(np.array([nir]), np.array([red]), 0.5)[0] == pytest.approx(expected)


def test_evi_formula():
    nir, red, blue = 0.5, 0.05, 0.03
    expected = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
    assert evi(np.array([nir]), np.array([red]), np.array([blue]))[0] == pytest.approx(expected)


def test_evi_out_of_range_becomes_nan():
    # A large blue value drives the denominator towards zero -> |EVI| > 1 -> undefined.
    value = evi(np.array([0.3]), np.array([0.05]), np.array([0.2]))
    assert np.isnan(value[0])


def test_zero_denominator_gives_nan_not_inf():
    result = ndvi(np.array([0.0, 0.4]), np.array([0.0, 0.2]))
    assert np.isnan(result[0])
    assert np.isfinite(result[1])
    assert not np.any(np.isinf(result))


def test_safe_divide_handles_nan_inputs():
    result = safe_divide(np.array([1.0, np.nan]), np.array([np.nan, 2.0]))
    assert np.all(np.isnan(result))


def test_ndvi_range_for_valid_reflectance():
    rng = np.random.default_rng(0)
    nir, red = rng.uniform(0, 1, 1000), rng.uniform(0, 1, 1000)
    values = ndvi(nir, red)
    finite = values[np.isfinite(values)]
    assert finite.min() >= -1 and finite.max() <= 1


def test_invalid_pixels_are_nan_in_every_index():
    bands = {b: np.full((4, 4), v) for b, v in
             zip(["blue", "green", "red", "red_edge", "nir"], [0.03, 0.08, 0.04, 0.3, 0.5])}
    mask = np.ones((4, 4), bool)
    mask[0, 0] = False
    indices, skipped = compute_indices(bands, mask)
    assert not skipped
    for array in indices.values():
        assert np.isnan(array[0, 0])
        assert np.isfinite(array[1, 1])


def test_missing_band_skips_dependent_indices():
    bands = {"red": np.array([0.1]), "nir": np.array([0.5]), "green": np.array([0.08])}
    indices, skipped = compute_indices(bands, np.array([True]))
    assert set(indices) == {"ndvi", "gndvi", "savi"}
    assert set(skipped) == {"ndre", "evi"}
    assert "red_edge" in skipped["ndre"]


def test_feature_extraction_order_and_values(make_sample):
    sample = make_sample("healthy")
    indices, _ = compute_indices(sample.bands, sample.valid_mask)
    features = extract_features(indices, sample.valid_mask)
    assert list(features.values) == FEATURE_ORDER
    assert not features.missing
    assert features.values["ndvi_mean"] == pytest.approx(np.nanmean(indices["ndvi"]))
    assert features.values["ndre_ndvi_ratio"] == pytest.approx(
        features.values["ndre_mean"] / features.values["ndvi_mean"])


def test_few_observations_produce_warning():
    bands = {"red": np.array([0.05, 0.06]), "nir": np.array([0.5, 0.45])}
    indices, _ = compute_indices(bands, np.array([True, True]))
    features = extract_features(indices, np.array([True, True]), min_observations=25)
    assert any("unreliable" in w for w in features.warnings)
