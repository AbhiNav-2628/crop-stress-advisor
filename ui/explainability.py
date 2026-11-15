import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.common import (
    DIVERGING_NEGATIVE, DIVERGING_POSITIVE, category_label, fmt, mode_banner, pick_result,
    require_results,
)


def _contribution_chart(rows: list[dict], target: str) -> go.Figure:
    rows = sorted(rows, key=lambda r: r["contribution"])
    fig = go.Figure(go.Bar(
        x=[r["contribution"] for r in rows],
        y=[r["feature"] for r in rows],
        orientation="h",
        marker_color=[DIVERGING_POSITIVE if r["contribution"] > 0 else DIVERGING_NEGATIVE for r in rows],
        customdata=[[fmt(r["value"], 4), r["zscore"]] for r in rows],
        hovertemplate="%{y}<br>value %{customdata[0]}<br>z-score %{customdata[1]:+.2f}"
                      "<br>contribution %{x:+.3f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_width=1, line_color="gray")
    fig.update_layout(
        title=f"Feature contributions to the score of '{category_label(target)}'",
        xaxis_title="contribution to class score (log-odds units)  ·  blue raises · red lowers",
        height=380, margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def render() -> None:
    st.title("Explainability")
    mode_banner()
    result = pick_result(require_results(), "xai_sample")
    explanation = result.explanation

    st.info(explanation["headline"])

    if explanation["inconclusive_reasons"]:
        st.subheader("Why the result is inconclusive")
        for reason in explanation["inconclusive_reasons"]:
            st.markdown(f"- {reason}")

    method = explanation["method"]
    if method == "linear_contributions":
        st.subheader("Model explanation: exact linear contributions")
        target = result.prediction.explanation["explained_class"]
        st.plotly_chart(_contribution_chart(explanation["contributions"], target))
        st.markdown(
            "**How to read this.** The model computes a score for each class as "
            "`intercept + Σ coefficient × z-score`, where the z-score says how far a feature is "
            "from the average *synthetic training plot*. Each bar is one term of that sum for the "
            "predicted class, so the bars add up exactly to the score (minus the intercept "
            f"{result.prediction.explanation['intercept']:+.3f}). This is not an approximation."
        )
        st.caption(
            "Caution: NDVI, SAVI, EVI and GNDVI are strongly correlated. With correlated inputs a "
            "linear model can give them large weights of opposite sign that largely cancel out, so "
            "a single bar should not be over-interpreted - look at groups of related features."
        )
        table = pd.DataFrame(explanation["contributions"])
        table["value"] = table["value"].map(lambda v: fmt(v, 4))
        table = table.round({"zscore": 2, "contribution": 3})
        st.dataframe(table[["feature", "value", "zscore", "contribution", "description"]], hide_index=True)
        with st.expander("Class scores (logits) for every class"):
            logits = result.prediction.explanation["logits"]
            st.dataframe(pd.DataFrame(
                [{"class": c, "score": round(v, 3)} for c, v in sorted(logits.items(), key=lambda i: -i[1])]
            ), hide_index=True)

    elif method == "rule_trace":
        st.subheader("Rule trace (checked in order, first fully matched rule decides)")
        fired = result.prediction.explanation.get("fired_rule")
        for rule in explanation["rule_trace"]:
            status = "🟢 FIRED" if rule["rule_id"] == fired else ("matched (later rule)" if rule["matched"] else "not matched")
            with st.expander(f"{rule['rule_id']} → {rule['category']} · {status}", expanded=rule["rule_id"] == fired):
                st.write(rule["description"])
                if rule["all_of"]:
                    st.markdown("**All of:**")
                    for c in rule["all_of"]:
                        st.markdown(f"- {'✔️ met' if c['passed'] else '✖️ not met'} · `{c['text']}`")
                if rule["any_of"]:
                    st.markdown("**At least one of:**")
                    for c in rule["any_of"]:
                        st.markdown(f"- {'✔️ met' if c['passed'] else '✖️ not met'} · `{c['text']}`")
        st.caption("Thresholds are defined in `config/settings.yaml` → `rule_classifier`.")
    else:
        st.write("No classifier explanation is available because the classifier was not run.")

    st.subheader("Context")
    st.markdown(f"- Mean NDVI reading (generic, crop-independent): **{explanation['ndvi_reading']}**.")
    for note in explanation["uncertainty_notes"]:
        st.markdown(f"- {note}")

    st.subheader("Limitations of this explanation")
    for item in explanation["limitations"]:
        st.markdown(f"- {item}")
    st.warning(
        "Attribution ≠ causation. A large contribution means the feature moved the model's score; "
        "it does not mean that this spectral property caused, or proves, a specific crop problem."
    )
