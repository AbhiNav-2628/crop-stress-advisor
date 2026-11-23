"""Drive the real Streamlit app headlessly with streamlit.testing.AppTest."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")

PAGES = [
    "Home", "Data Input", "Vegetation Indices", "Crop-Stress Assessment", "Explainability",
    "Advisory", "Evaluation", "Reports", "About & Responsible AI",
]


def new_app() -> AppTest:
    return AppTest.from_file(APP_PATH, default_timeout=120).run()


def go(app: AppTest, page: str) -> AppTest:
    app.sidebar.radio(key="page").set_value(page).run()
    assert not app.exception, [e.value for e in app.exception]
    return app


def test_app_starts_without_errors():
    app = new_app()
    assert not app.exception
    assert "SIH25099" in " ".join(m.value for m in app.markdown)


def test_pages_show_empty_state_before_data():
    app = new_app()
    for page in ["Vegetation Indices", "Crop-Stress Assessment", "Reports"]:
        go(app, page)
        assert any("No data loaded" in i.value for i in app.info)


@pytest.mark.parametrize("mode", ["ml_demo", "cnn_deep_learning", "rule_based"])
def test_full_demo_workflow(mode):
    app = new_app()
    app.sidebar.selectbox(key="mode").set_value(mode).run()
    app.button[0].click().run()  # Home: "Load demo sample and analyse"
    assert app.session_state.page == "Crop-Stress Assessment"
    assert app.session_state.results[0].prediction.category == "nutrient_stress"
    for page in PAGES:
        go(app, page)

    go(app, "Reports")
    app.button[0].click().run()  # Generate report
    assert not app.exception
    assert app.session_state.report["samples"][0]["prediction"]["category"] == "nutrient_stress"
    assert app.session_state.report["classifier"]["mode"] == mode


def test_load_bundled_csv_with_multiple_samples():
    app = new_app()
    go(app, "Data Input")
    choice = app.selectbox(key="demo_choice")
    # Options are rendered with a format function, so select by position.
    choice.select_index(choice.options.index("demo_points.csv")).run()
    next(b for b in app.button if b.label == "Load selected demo file").click().run()
    assert not app.exception
    assert len(app.session_state.samples) == 5
    go(app, "Crop-Stress Assessment")
    go(app, "Evaluation")


def test_broken_file_shows_validation_error():
    app = new_app()
    go(app, "Data Input")
    next(b for b in app.button if b.label == "Validate broken file").click().run()
    assert any("Validation failed" in e.value for e in app.error)


def test_deep_link_loads_demo_and_page():
    app = AppTest.from_file(APP_PATH, default_timeout=120)
    app.query_params["demo"] = "plot_pest_stress.tif"
    app.query_params["page"] = "Explainability"
    app.query_params["mode"] = "rule_based"
    app.run()
    assert not app.exception
    assert app.session_state.page == "Explainability"
    assert app.session_state.results[0].prediction.category == "pest_stress"
    assert app.session_state.results[0].prediction.classifier["mode"] == "rule_based"
