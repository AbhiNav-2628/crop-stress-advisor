"""Shared helpers for the Streamlit pages: session state, cached resources,
running the analysis and small display components."""

import math

import streamlit as st

from src.config import load_advisory_rules, load_settings
from src.models.inference import MODE_LABELS, build_classifier
from src.pipeline import analyze_sample

# Chart colours (from a colour-blind-validated reference palette).
SERIES_BLUE = "#2a78d6"
DIVERGING_POSITIVE = "#2a78d6"   # raises the class score
DIVERGING_NEGATIVE = "#e34948"   # lowers the class score
MUTED_GRAY = "#a8a69e"
MISSING_PIXEL_GRAY = "#8f8d86"

# Status styling for result badges: always icon + label, never colour alone.
CATEGORY_STYLE = {
    "healthy": ("✅", "#0ca30c"),
    "water_stress": ("⚠️", "#ec835a"),
    "nutrient_stress": ("⚠️", "#ec835a"),
    "pest_stress": ("⚠️", "#ec835a"),
    "disease_stress": ("⚠️", "#ec835a"),
    "inconclusive": ("❔", "#8f8d86"),
}
SEVERITY_ICON = {"info": "ℹ️", "low": "🔹", "moderate": "🔸", "high": "🔺"}


@st.cache_resource
def get_settings() -> dict:
    return load_settings()


@st.cache_resource
def get_rules() -> list[dict]:
    return load_advisory_rules()


@st.cache_resource
def get_classifier(mode: str):
    return build_classifier(mode, get_settings())


def init_state() -> None:
    ss = st.session_state
    ss.setdefault("samples", [])
    ss.setdefault("input_label", None)
    ss.setdefault("mode", get_settings()["model"]["default_mode"])
    ss.setdefault("crop", "generic")
    ss.setdefault("results", [])
    ss.setdefault("results_key", None)
    ss.setdefault("selected_index", 0)


def category_label(category: str) -> str:
    return get_settings()["categories"].get(category, category)


def current_results() -> list:
    """Analyse the loaded samples with the active mode/crop, re-running only when needed."""
    ss = st.session_state
    key = (id(ss.samples), ss.mode, ss.crop)
    if ss.results_key != key:
        classifier, _ = get_classifier(ss.mode)
        ss.results = [
            analyze_sample(sample, classifier, get_settings(), get_rules(),
                           sample.metadata.get("crop", ss.crop) if ss.crop == "generic" else ss.crop)
            for sample in ss.samples
        ]
        ss.results_key = key
    return ss.results


def set_samples(samples: list, label: str) -> None:
    ss = st.session_state
    ss.samples = samples
    ss.input_label = label
    ss.selected_index = 0
    ss.results_key = None


def pick_result(results: list, key: str):
    """Sample selector shown when several samples are loaded. Returns the chosen result."""
    if len(results) == 1:
        return results[0]
    labels = [f"{r.sample_id}  →  {r.prediction.category}" for r in results]
    index = st.selectbox(
        "Sample", range(len(results)), format_func=lambda i: labels[i],
        index=min(st.session_state.selected_index, len(results) - 1), key=key,
    )
    st.session_state.selected_index = index
    return results[index]


def require_results():
    """Return results, or show an empty state and stop the page."""
    results = current_results()
    if not results:
        st.info(
            "No data loaded yet. Go to **Data Input** and choose a bundled demonstration "
            "sample (no upload needed), or use the button on the Home page."
        )
        st.stop()
    return results


def mode_banner() -> None:
    classifier, notices = get_classifier(st.session_state.mode)
    for notice in notices:
        st.warning(notice)
    st.caption(
        f"**Active classifier:** {MODE_LABELS.get(classifier.mode, classifier.mode)} · "
        f"version `{classifier.version}` · "
        "demonstration system, not validated on field data."
    )


def category_badge(category: str, confidence: float | None = None) -> None:
    icon, color = CATEGORY_STYLE.get(category, ("•", MUTED_GRAY))
    conf = f" · top probability {confidence:.0%}" if confidence is not None else ""
    st.markdown(
        f"<div style='border-left:6px solid {color};padding:0.6rem 1rem;border-radius:6px;"
        f"background:rgba(128,128,128,0.08)'><span style='font-size:1.45rem;font-weight:600'>"
        f"{icon} {category_label(category)}</span><br><span style='opacity:0.8'>"
        f"category code: <code>{category}</code>{conf}</span></div>",
        unsafe_allow_html=True,
    )


def fmt(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)
