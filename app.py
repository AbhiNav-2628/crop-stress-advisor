"""Streamlit entry point.

Run with:   streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="CropStress Advisor · SIH25099",
    page_icon="🌾",
    layout="wide",
)

from src.data.validation import ValidationError  # noqa: E402
from src.models.inference import MODE_LABELS  # noqa: E402
from ui import (  # noqa: E402
    about, advisory, assessment, data_input, evaluation, explainability, home, indices, reports,
)
from ui.common import get_classifier, get_settings, init_state, set_samples  # noqa: E402
from ui.data_input import load_demo_file  # noqa: E402

PAGES = {
    "Home": home.render,
    "Data Input": data_input.render,
    "Vegetation Indices": indices.render,
    "Crop-Stress Assessment": assessment.render,
    "Explainability": explainability.render,
    "Advisory": advisory.render,
    "Evaluation": evaluation.render,
    "Reports": reports.render,
    "About & Responsible AI": about.render,
}


def sidebar() -> str:
    settings = get_settings()
    with st.sidebar:
        st.markdown(f"### 🌾 {settings['project']['short_name']}")
        st.caption(f"SIH 2025 · {settings['project']['problem_id']}")
        page = st.radio("Navigate", list(PAGES), key="page", label_visibility="collapsed")

        st.divider()
        st.selectbox("Classifier mode", list(MODE_LABELS), format_func=MODE_LABELS.get, key="mode")
        st.selectbox(
            "Crop", settings["crops"], key="crop",
            help="Used by crop-specific advisory rules. 'generic' keeps any crop given in a CSV file.",
        )
        classifier, notices = get_classifier(st.session_state.mode)
        if notices:
            st.warning("ML model unavailable - using rule-based fallback.")
        st.caption(f"Active: `{classifier.version}`")

        st.divider()
        if st.session_state.input_label:
            st.caption(f"Loaded: {st.session_state.input_label} ({len(st.session_state.samples)} sample(s))")
        else:
            st.caption("No data loaded yet.")
    return page


def apply_query_params() -> None:
    """Optional deep links, read once per session, e.g.
    http://localhost:8501/?demo=plot_pest_stress.tif&page=Explainability&mode=rule_based
    """
    ss = st.session_state
    if ss.get("query_applied"):
        return
    ss.query_applied = True
    params = st.query_params
    if params.get("mode") in MODE_LABELS:
        ss.mode = params["mode"]
    if params.get("page") in PAGES:
        ss.page = params["page"]
    if params.get("demo"):
        try:
            set_samples(load_demo_file(params["demo"]), f"Bundled demo: {params['demo']}")
        except (ValidationError, OSError) as error:
            st.error(f"Could not load demo '{params['demo']}': {error}")


def main() -> None:
    init_state()
    st.session_state.setdefault("page", "Home")
    apply_query_params()
    page = sidebar()
    PAGES[page]()


main()
