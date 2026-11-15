import streamlit as st

from ui.common import get_settings, set_samples
from ui.data_input import load_demo_file

WORKFLOW_STEPS = [
    "Multispectral input<br><small>GeoTIFF · band files · CSV</small>",
    "Validation &amp;<br>band mapping",
    "Vegetation indices<br><small>NDVI NDRE GNDVI SAVI EVI</small>",
    "Feature<br>engineering",
    "Stress<br>classification",
    "Uncertainty &amp;<br>quality gating",
    "Explanation",
    "Rule-based<br>advisories",
    "Human-reviewed<br>report",
]


def workflow_html() -> str:
    """The pipeline as plain HTML boxes (no external renderer needed, works offline)."""
    box = ("display:inline-block;padding:0.45rem 0.6rem;margin:0.2rem 0;border:1px solid rgba(128,128,128,0.45);"
           "border-radius:8px;background:rgba(47,143,58,0.08);text-align:center;font-size:0.85rem;"
           "line-height:1.25;vertical-align:middle")
    arrow = "<span style='margin:0 0.3rem;opacity:0.6'>&#8594;</span>"
    return arrow.join(f"<span style='{box}'>{step}</span>" for step in WORKFLOW_STEPS)


def load_quick_demo() -> None:
    set_samples(load_demo_file("plot_nutrient_stress.tif"), "Bundled demo: plot_nutrient_stress.tif")
    # Runs as a button callback, i.e. before the sidebar radio is drawn, so it may change the page.
    st.session_state.page = "Crop-Stress Assessment"


def render() -> None:
    project = get_settings()["project"]
    st.title("🌾 " + project["name"])
    st.markdown(f"**{project['event']} · Problem ID {project['problem_id']}**")

    st.warning(
        "**Demonstration system.** The classifiers were fitted to or tuned on clearly labelled "
        "*synthetic* data. Outputs are prompts for field inspection, not validated agronomic "
        "diagnoses. No pesticide, fertiliser or treatment is ever prescribed."
    )

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Problem statement")
        st.write(
            "Crop stress caused by water deficit, nutrient deficiency, pests or disease often "
            "becomes visible to the eye only after yield is already affected. Multispectral "
            "sensors (UAV or satellite) record red-edge and near-infrared reflectance that "
            "change earlier. Farmers and extension workers need a tool that turns these "
            "measurements into understandable, cautious, verifiable advice."
        )
        st.subheader("Objectives")
        st.markdown(
            "- Validate multispectral inputs and map sensor bands correctly.\n"
            "- Compute standard vegetation indices with safe handling of invalid pixels.\n"
            "- Classify potential stress categories through a swappable model interface.\n"
            "- Quantify uncertainty and say **inconclusive** when evidence is weak.\n"
            "- Explain each result with the actual numbers and rules behind it.\n"
            "- Produce responsible, rule-based advisories that call for human verification.\n"
            "- Export a traceable report (JSON, CSV, PDF)."
        )
    with right:
        st.subheader("Try it in one click")
        st.write("Loads a bundled synthetic plot. No upload, training or internet needed.")
        st.button("▶ Load demo sample and analyse", type="primary", width="stretch",
                  on_click=load_quick_demo)
        st.subheader("Operating modes")
        st.markdown(
            "**1. Demonstration ML model (default)** - multinomial logistic regression "
            "fitted to synthetic plots; coefficients bundled as JSON.\n\n"
            "**2. Transparent rule-based baseline** - an ordered decision list with "
            "thresholds in `config/settings.yaml`.\n\n"
            "If the ML artifact is missing or damaged, the app falls back to mode 2 automatically "
            "and says so. Switch modes in the sidebar."
        )

    st.subheader("System workflow")
    st.markdown(workflow_html(), unsafe_allow_html=True)
    st.caption(
        "Use the sidebar to move through the steps: Data Input → Vegetation Indices → "
        "Assessment → Explainability → Advisory → Reports."
    )
