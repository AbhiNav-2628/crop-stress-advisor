import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import resolve_path
from src.data.ingestion import RasterInput
from src.data.preprocessing import sample_from_raster
from src.data.synthetic import SCENARIOS, generate_dataset
from src.models.artifacts import load_metrics
from src.models.classifier import INCONCLUSIVE
from src.models.evaluation import classification_metrics
from src.models.inference import predict_with_uncertainty
from src.pipeline import compute_sample_features
from ui.common import SERIES_BLUE, current_results, get_classifier, get_settings

CONFIG_LABELS = {
    "ml_raw": "ML model, raw argmax (no gating)",
    "ml_with_uncertainty_gating": "ML model with uncertainty gating",
    "rule_based_with_gating": "Rule-based baseline with gating",
}


def _summary_table(metrics: dict) -> pd.DataFrame:
    rows = []
    for key, label in CONFIG_LABELS.items():
        if key not in metrics:
            continue
        m = metrics[key]
        rows.append({
            "configuration": label,
            "accuracy (inconclusive = wrong)": round(m["accuracy"], 3),
            "macro F1": round(m["macro_f1"], 3),
            "coverage (share decided)": round(m["coverage"], 3),
            "accuracy on decided": None if m["accuracy_on_decided"] is None else round(m["accuracy_on_decided"], 3),
            "n": m["n_samples"],
        })
    return pd.DataFrame(rows)


def _confusion_figure(m: dict, title: str) -> go.Figure:
    labels = m["confusion_matrix"]["labels"]
    true_labels = [l for l in labels if l != INCONCLUSIVE]
    matrix = np.array(m["confusion_matrix"]["matrix"])[: len(true_labels)]
    fig = px.imshow(matrix, x=labels, y=true_labels, text_auto=True, color_continuous_scale="Blues",
                    labels=dict(x="predicted", y="true (synthetic)", color="count"), aspect="auto")
    fig.update_layout(title=title, height=380, margin=dict(l=10, r=10, t=40, b=10), coloraxis_showscale=False)
    return fig


def _per_class_table(m: dict) -> pd.DataFrame:
    return pd.DataFrame([
        {"class": c, **{k: round(v, 3) if isinstance(v, float) else v for k, v in vals.items()}}
        for c, vals in m["per_class"].items()
    ])


def _importance_chart(importance: dict) -> go.Figure:
    items = sorted(importance.items(), key=lambda i: i[1]["mean"])
    fig = go.Figure(go.Bar(
        x=[v["mean"] for _, v in items], y=[k for k, _ in items], orientation="h",
        marker_color=SERIES_BLUE,
        error_x=dict(type="data", array=[v["std"] for _, v in items], color="gray"),
        hovertemplate="%{y}: accuracy drop %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(title="Permutation importance on the synthetic test set (drop in accuracy)",
                      height=360, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="mean accuracy drop")
    return fig


@st.cache_data(show_spinner="Generating fresh synthetic plots and evaluating...")
def fresh_synthetic_evaluation(n_per_class: int, seed: int) -> dict:
    settings = get_settings()
    mapping = next(iter(settings["band_presets"].values()))
    ml, _ = get_classifier("ml_demo")
    rules, _ = get_classifier("rule_based")
    y_true, ml_pred, rule_pred = [], [], []
    for i, (scenario, cube) in enumerate(generate_dataset(n_per_class, seed)):
        sample = sample_from_raster(RasterInput(cube, None, f"eval-{i}", "synthetic"), mapping, settings)
        _, _, features = compute_sample_features(sample, settings)
        y_true.append(scenario)
        ml_pred.append(predict_with_uncertainty(features, ml, settings).category)
        rule_pred.append(predict_with_uncertainty(features, rules, settings).category)
    return {
        "ml_with_uncertainty_gating": classification_metrics(y_true, ml_pred, SCENARIOS),
        "rule_based_with_gating": classification_metrics(y_true, rule_pred, SCENARIOS),
    }


def render() -> None:
    st.title("Evaluation")
    st.warning(
        "**No field validation has been performed.** There are no accuracy figures for real crops. "
        "Everything on this page is computed on *synthetic* plots from the same generator the "
        "model was fitted to. It shows that the software works as designed and illustrates "
        "uncertainty trade-offs - it is **not** evidence of real-world performance."
    )

    settings = get_settings()
    metrics = load_metrics(resolve_path(settings["model"]["metrics_path"]))
    st.subheader("1 · Held-out synthetic test set (saved when the demo model was fitted)")
    if metrics is None:
        st.warning("Metrics file not found. Run `python scripts/train_demo_model.py` to recreate it.")
    else:
        st.caption(f"{metrics['evaluation_data']} · created {metrics['created_at']} · n = {metrics['n_test']}")
        st.dataframe(_summary_table(metrics), hide_index=True)
        st.markdown(
            "*Coverage* is the share of plots the system was willing to classify. Uncertainty gating "
            "trades coverage for reliability: it abstains on hard cases, so accuracy on the decided "
            "plots rises."
        )
        choice = st.radio("Details for", list(CONFIG_LABELS), format_func=CONFIG_LABELS.get, horizontal=True)
        left, right = st.columns(2)
        with left:
            st.plotly_chart(_confusion_figure(metrics[choice], "Confusion matrix (synthetic test set)"))
        with right:
            st.dataframe(_per_class_table(metrics[choice]), hide_index=True)
            st.caption("The pest vs disease confusion is expected: both appear as patchy vigour loss.")
        st.plotly_chart(_importance_chart(metrics["permutation_importance"]))
        st.caption("Permutation importance is a global, model-level measure on synthetic data; it says "
                   "which inputs this model relies on, not which spectral properties matter biologically.")

    st.subheader("2 · Run a fresh evaluation now")
    st.write("Generates new synthetic plots with a seed of your choice and evaluates both classifiers live.")
    c1, c2 = st.columns(2)
    n = c1.slider("Plots per class", 20, 200, 60, step=20)
    seed = c2.number_input("Random seed", 0, 10_000, 101)
    if st.button("Run evaluation", type="primary"):
        live = fresh_synthetic_evaluation(int(n), int(seed))
        st.dataframe(_summary_table(live), hide_index=True)
        st.plotly_chart(_confusion_figure(live["ml_with_uncertainty_gating"], "ML with gating - fresh synthetic plots"))

    st.subheader("3 · Evaluate on the loaded labelled samples")
    labelled = [r for r in current_results()
                if r.sample.metadata.get("label") or r.sample.metadata.get("synthetic_label")]
    if not labelled:
        st.write(
            "Load a CSV whose rows include a `label` column (one of "
            f"{', '.join(SCENARIOS)}) to compute metrics on your own data. The bundled "
            "`demo_points.csv` carries synthetic labels."
        )
    else:
        y_true = [r.sample.metadata.get("label") or r.sample.metadata.get("synthetic_label") for r in labelled]
        y_pred = [r.prediction.category for r in labelled]
        m = classification_metrics(y_true, y_pred, SCENARIOS)
        st.write(f"{len(labelled)} labelled sample(s) loaded. With so few samples these numbers are anecdotal.")
        st.dataframe(pd.DataFrame({"sample": [r.sample_id for r in labelled], "label": y_true, "predicted": y_pred}),
                     hide_index=True)
        st.metric("Accuracy on loaded labelled samples", f"{m['accuracy']:.2f}")
