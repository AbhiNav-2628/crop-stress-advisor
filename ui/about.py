import streamlit as st

from src import __version__
from src.config import resolve_path
from src.models.artifacts import ArtifactError, load_artifact
from ui.common import get_settings


def render() -> None:
    st.title("About & Responsible AI")
    settings = get_settings()

    st.subheader("Data provenance")
    st.markdown(
        "- **All bundled data is synthetic.** It is produced by `src/data/synthetic.py` using a linear "
        "mixing model of illustrative leaf and soil reflectance spectra, with seeded random "
        "variation. No field measurements, farmer data or third-party datasets are included.\n"
        "- The demo files use fixed, clearly expressed stress severities so each is a readable "
        "example; training and evaluation data use random severities and nuisance variation."
    )
    try:
        artifact = load_artifact(resolve_path(settings["model"]["artifact_path"]))
        st.markdown(
            f"- **Demonstration model:** {artifact['name']} `{artifact['version']}`, "
            f"created {artifact.get('created_at', 'n/a')}, scikit-learn {artifact.get('sklearn_version', 'n/a')}. "
            f"{artifact['provenance']}"
        )
    except ArtifactError as error:
        st.warning(f"Model artifact unavailable: {error}")

    st.subheader("Methodology (summary)")
    st.markdown(
        "1. Validate input files, band mapping and reflectance scale; mask invalid pixels.\n"
        "2. Compute NDVI, NDRE, GNDVI, SAVI and EVI with safe division.\n"
        "3. Summarise into 9 plot-level features (means, spatial spread, low-vigour share, NDRE/NDVI).\n"
        "4. Classify with the demo logistic regression or the rule-based decision list.\n"
        "5. Apply quality, out-of-distribution and confidence gates → possibly *inconclusive*.\n"
        "6. Explain with exact linear contributions or the rule trace.\n"
        "7. Match independent advisory rules; every advisory asks for human verification.\n\n"
        "Details: `docs/methodology.md`, `docs/model_card.md`, `docs/architecture.md`."
    )

    st.subheader("Limitations")
    st.markdown(
        "- Not validated on real fields; no accuracy claim for real crops is made.\n"
        "- Vegetation indices cannot reliably separate pest from disease, or water from nutrient "
        "stress, in general. The synthetic evaluation already shows pest/disease confusion.\n"
        "- Thresholds are not calibrated for any crop, variety, growth stage or sensor.\n"
        "- Very mild stress can be missed (see `plot_mild_ambiguous.tif`).\n"
        "- No thermal, soil-moisture, weather or field-history data is used.\n"
        "- Full list: `docs/limitations.md`."
    )

    st.subheader("Privacy and security")
    st.markdown(
        "- Everything runs locally. Uploaded files are processed in memory and are not stored or sent anywhere.\n"
        "- No API keys, accounts or cloud services. Streamlit usage statistics are disabled in `.streamlit/config.toml`.\n"
        "- The model is stored as JSON, not pickle, so loading it cannot execute code.\n"
        "- GeoTIFFs may contain location coordinates; treat exported reports as potentially location-sensitive."
    )

    st.subheader("Responsible-use statement")
    st.info(
        "This system supports, and never replaces, human judgement. Use it to decide *where to look*, "
        "not *what to spray or apply*. Confirm every finding in the field and consult a qualified "
        "agricultural expert (e.g. a KVK or extension officer) before acting. Do not use it for "
        "insurance, credit, compensation or regulatory decisions."
    )

    st.subheader("Project")
    st.markdown(
        f"- {settings['project']['name']} · {settings['project']['event']} · Problem ID "
        f"{settings['project']['problem_id']} · app version {__version__}\n"
        "- Documentation: `README.md` and the `docs/` folder (architecture, methodology, model card, "
        "limitations, syllabus mapping, learning guide, report outline, test results)."
    )
