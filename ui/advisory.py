import streamlit as st

from ui.common import SEVERITY_ICON, category_label, fmt, mode_banner, pick_result, require_results


def render() -> None:
    st.title("Advisory Recommendations")
    mode_banner()
    result = pick_result(require_results(), "adv_sample")
    prediction = result.prediction
    features = result.features.values

    st.caption(
        "Observations, predictions and recommendations are kept separate on purpose. Advisories "
        "come from an independent rule engine (`config/advisory_rules.yaml`) and never prescribe "
        "pesticides, fertiliser doses or other treatments."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1 · Observations (measured)")
        st.markdown(
            f"- Valid observations: **{result.features.n_valid}/{result.features.n_total}** "
            f"({result.features.valid_fraction:.0%})\n"
            f"- Mean NDVI **{fmt(features['ndvi_mean'])}**, NDRE **{fmt(features['ndre_mean'])}**, "
            f"GNDVI **{fmt(features['gndvi_mean'])}**\n"
            f"- NDVI spatial std **{fmt(features['ndvi_std'])}**, 10th percentile **{fmt(features['ndvi_p10'])}**\n"
            f"- Data-quality warnings: **{len(result.warnings)}**\n"
            f"- Crop context: **{result.crop}**"
        )
    with col2:
        st.subheader("2 · Prediction (model output)")
        confidence = "" if prediction.confidence is None else f" (top probability {prediction.confidence:.2f})"
        st.markdown(f"- Category: **{category_label(prediction.category)}**{confidence}")
        st.markdown(f"- Classifier: {prediction.classifier.get('name')} `{prediction.classifier.get('version')}`")
        if prediction.is_inconclusive:
            st.markdown("- Withheld as inconclusive - see Assessment page.")

    st.subheader(f"3 · Recommendations ({len(result.advisories)} matching rules)")
    for advisory in result.advisories:
        icon = SEVERITY_ICON.get(advisory.severity, "•")
        with st.container(border=True):
            st.markdown(f"#### {icon} {advisory.title}")
            st.caption(
                f"Rule `{advisory.rule_id}` · severity **{advisory.severity}** · priority {advisory.priority}"
                + (" · illustrative rule, needs expert review" if advisory.illustrative else "")
            )
            st.write(advisory.message)
            st.markdown(f"**Human verification step:** {advisory.verification}")
            if advisory.evidence:
                evidence = ", ".join(f"`{k}` = {fmt(v)}" for k, v in advisory.evidence.items())
                st.markdown(f"**Evidence:** {evidence}")
            if advisory.matched_conditions:
                st.markdown("**Activated because:** " + "; ".join(f"`{c}`" for c in advisory.matched_conditions))
            if advisory.limitations and advisory.limitations != "-":
                st.markdown(f"**Limitations:** {advisory.limitations}")
            st.caption(f"Source: {advisory.source}")

    st.subheader("Next steps checklist")
    steps = list(dict.fromkeys(a.verification for a in result.advisories))
    for i, step in enumerate(steps, 1):
        st.checkbox(step, key=f"step_{result.sample_id}_{i}")
