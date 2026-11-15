import io
import json

import numpy as np
import pandas as pd
import pytest

from src.data.ingestion import RasterInput
from src.data.preprocessing import sample_from_raster
from src.data.synthetic import add_cloud_and_nodata, synthesize_plot
from src.features.feature_engineering import FEATURE_ORDER
from src.models.inference import build_classifier
from src.pipeline import analyze_sample
from src.reporting.report_generator import build_report, report_to_csv, report_to_json, report_to_pdf

DEMO_MAPPING = {"blue": 1, "green": 2, "red": 3, "red_edge": 4, "nir": 5}


@pytest.fixture(scope="module")
def report(settings, advisory_rules):
    """A two-sample report: one clear water-stress plot and one mostly-nodata plot."""
    rng = np.random.default_rng(5)
    cubes = {
        "water": synthesize_plot("water_stress", rng, severity=0.8, typical=True),
        "cloudy": add_cloud_and_nodata(synthesize_plot("healthy", rng, typical=True), rng, 0.7),
    }
    classifier, notices = build_classifier("ml_demo", settings)
    results = []
    for name, cube in cubes.items():
        sample = sample_from_raster(RasterInput(cube, -9999.0, f"{name}.tif", "test"),
                                    DEMO_MAPPING, settings, 1.0, name)
        results.append(analyze_sample(sample, classifier, settings, advisory_rules, "wheat"))
    return build_report(results, classifier.info(), notices, settings)


def test_report_has_required_sections(report):
    for key in ("report_id", "generated_at", "system", "classifier", "disclaimer", "samples"):
        assert key in report
    sample = report["samples"][0]
    for key in ("input", "vegetation_indices", "features", "prediction", "explanation",
                "data_quality_warnings", "advisories"):
        assert key in sample
    assert report["classifier"]["mode"] == "ml_demo"
    assert report["classifier"]["version"]


def test_json_round_trip_has_no_nan(report):
    text = report_to_json(report)
    assert "NaN" not in text
    assert json.loads(text)["report_id"] == report["report_id"]


def test_inconclusive_sample_reported_with_reasons(report):
    cloudy = report["samples"][1]
    assert cloudy["prediction"]["category"] == "inconclusive"
    assert cloudy["prediction"]["inconclusive_reasons"]
    assert any(a["rule_id"] == "ADV-INC-02" for a in cloudy["advisories"])


def test_csv_has_one_row_per_sample(report):
    frame = pd.read_csv(io.StringIO(report_to_csv(report)))
    assert len(frame) == 2
    assert set(FEATURE_ORDER).issubset(frame.columns)
    assert frame.loc[0, "category"] == "water_stress"
    assert frame.loc[1, "is_inconclusive"]


def test_pdf_is_generated(report):
    pdf = report_to_pdf(report)
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 2000


def test_report_ids_are_unique(settings):
    a = build_report([], {"mode": "x"}, [], settings)
    b = build_report([], {"mode": "x"}, [], settings)
    assert a["report_id"] != b["report_id"]
