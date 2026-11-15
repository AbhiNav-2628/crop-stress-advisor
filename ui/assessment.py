import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.features.feature_engineering import FEATURE_DESCRIPTIONS, FEATURE_ORDER
from ui.common import (
    MUTED_GRAY, SERIES_BLUE, category_badge, category_label, fmt, mode_banner, pick_result,
    require_results,
)


def _probability_chart(probabilities: dict, predicted: str) -> go.Figure:
    items = sorted(probabilities.items(), key=lambda item: item[1])
    labels = [category_label(c) for c, _ in items]
    values = [p for _, p in items]
    colors = [SERIES_BLUE if c == predicted else MUTED_GRAY for c, _ in items]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker_color=colors,
        text=[f"{v:.0%}" for v in values], textposition="outside",
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title="Class probabilities from the demonstration model", height=300,
        xaxis=dict(range=[0, 1.12], tickformat=".0%", title="probability"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def _overview_table(results: list) -> pd.DataFrame:
    rows = []
    for r in results:
        p = r.prediction
        rows.append({
            "sample": r.sample_id,
            "category": p.category,
            "classifier said": p.raw_category or "(not run)",
            "top probability": None if p.confidence is None else round(p.confidence, 3),
            "valid %": round(100 * r.features.valid_fraction, 1),
            "NDVI mean": round(r.features.values["ndvi_mean"], 3),
            "advisories": len(r.advisories),
            "synthetic label (demo only)": r.sample.metadata.get("synthetic_label", ""),
        })
    return pd.DataFrame(rows)


def render() -> None:
    st.title("Crop-Stress Assessment")
    mode_banner()
    results = require_results()

    if len(results) > 1:
        st.subheader("All samples")
        st.dataframe(_overview_table(results), hide_index=True)
        st.divider()

    result = pick_result(results, "assess_sample")
    prediction = result.prediction

    category_badge(prediction.category, None if prediction.is_inconclusive else prediction.confidence)
    st.write("")

    if prediction.is_inconclusive:
        st.warning("**Inconclusive.** The system declined to assign a category because:")
        for reason in prediction.reasons:
            st.markdown(f"- {reason}")
        if prediction.raw_category and prediction.raw_category != "inconclusive":
            st.caption(f"For transparency: the classifier's withheld raw answer was `{prediction.raw_category}`.")

    label = result.sample.metadata.get("synthetic_label") or result.sample.metadata.get("label")
    if label:
        st.info(
            f"Ground-truth label stored in this demo file: **{label}**. It is shown only so you can "
            "compare; it was not used by the classifier."
        )

    left, right = st.columns([3, 2])
    with left:
        if prediction.probabilities:
            st.plotly_chart(_probability_chart(prediction.probabilities, prediction.raw_category))
            st.caption(
                "Probabilities come from a model fitted to synthetic data. They are not calibrated "
                "for real fields and must not be read as the chance that the crop is stressed."
            )
        elif prediction.raw_category is None:
            st.markdown("The classifier was **not run** because the input failed the quality checks above.")
        else:
            st.markdown(
                "The rule-based classifier does not produce probabilities. Its decision is fully "
                "described by the rule trace on the **Explainability** page."
            )
    with right:
        st.subheader("Uncertainty assessment")
        for note in prediction.uncertainty_notes:
            st.markdown(f"- {note}")
        if not prediction.uncertainty_notes:
            st.markdown("- No additional notes.")
        settings_note = (
            "Gating rules: inconclusive if valid pixels < 50%, required features missing, "
            "mean NDVI < 0.3 (non-crop), top probability < 0.60, margin < 0.15, or any feature "
            "beyond ±4 training standard deviations (thresholds in `config/settings.yaml`)."
        )
        st.caption(settings_note)

    st.subheader("Supporting evidence: engineered features")
    st.dataframe(pd.DataFrame([
        {"feature": name, "value": fmt(result.features.values.get(name), 4),
         "meaning": FEATURE_DESCRIPTIONS[name]}
        for name in FEATURE_ORDER
    ]), hide_index=True)

    if result.warnings:
        st.subheader("Data-quality warnings")
        for warning in result.warnings:
            st.warning(warning)

    st.caption(
        "This result is a demonstration output and does not represent a validated agricultural "
        "diagnosis. See **Advisory** for the recommended verification steps."
    )
