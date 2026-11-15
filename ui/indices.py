import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.features.vegetation_indices import INDEX_FORMULAS, INDEX_LABELS
from ui.common import MISSING_PIXEL_GRAY, SERIES_BLUE, pick_result, require_results


def _true_colour(sample) -> np.ndarray | None:
    """Simple 2-98% stretched RGB composite for orientation (display only)."""
    if not all(b in sample.bands for b in ("red", "green", "blue")):
        return None
    rgb = np.dstack([sample.bands["red"], sample.bands["green"], sample.bands["blue"]])
    rgb = np.where(sample.valid_mask[..., None], rgb, np.nan)
    low, high = np.nanpercentile(rgb, 2), np.nanpercentile(rgb, 98)
    stretched = np.clip((rgb - low) / (high - low + 1e-9), 0, 1)
    stretched = np.nan_to_num(stretched, nan=0.55)  # invalid pixels shown grey
    return (stretched * 255).astype("uint8")


def _index_map(name: str, array: np.ndarray) -> go.Figure:
    fig = px.imshow(
        array, color_continuous_scale="Greens", zmin=0 if name != "evi" else -0.1, zmax=1,
        aspect="equal", labels={"color": name.upper()},
    )
    fig.update_layout(
        plot_bgcolor=MISSING_PIXEL_GRAY, margin=dict(l=10, r=10, t=30, b=10), height=420,
        title=f"{name.upper()} map (grey = invalid / undefined pixels)",
    )
    fig.update_xaxes(showticklabels=False, showgrid=False)
    fig.update_yaxes(showticklabels=False, showgrid=False)
    fig.update_traces(hovertemplate="row %{y}, col %{x}<br>value %{z:.3f}<extra></extra>")
    return fig


def _histogram(name: str, array: np.ndarray) -> go.Figure:
    finite = array[np.isfinite(array)]
    fig = go.Figure(go.Histogram(x=finite, nbinsx=30, marker_color=SERIES_BLUE,
                                 marker_line_width=1, marker_line_color="rgba(255,255,255,0.6)",
                                 hovertemplate="%{x}<br>%{y} observations<extra></extra>"))
    fig.update_layout(
        title=f"Distribution of {name.upper()} (valid observations only)", height=300,
        margin=dict(l=10, r=10, t=40, b=10), xaxis_title=name.upper(), yaxis_title="count",
        bargap=0.05,
    )
    return fig


def render() -> None:
    st.title("Vegetation-Index Analysis")
    result = pick_result(require_results(), "idx_sample")
    sample = result.sample

    st.markdown(
        f"**{sample.sample_id}** · {sample.kind} · valid observations "
        f"**{sample.n_valid}/{sample.n_total} ({sample.valid_fraction:.0%})**"
    )
    for name, reason in result.skipped_indices.items():
        st.warning(f"{name.upper()} was not calculated: {reason}.")

    st.subheader("Summary statistics (valid values only)")
    st.dataframe(result.index_summary.round(4), hide_index=True)
    st.caption(
        "`undefined_values` counts observations that were invalid in the input or whose index "
        "denominator was zero; they are excluded from every statistic, never replaced by guesses."
    )

    if not result.indices:
        st.info("No index could be calculated.")
        return

    if sample.kind == "raster":
        st.subheader("Raster layers")
        names = list(result.indices)
        tabs = st.tabs([n.upper() for n in names] + ["Validity mask", "True colour"])
        for tab, name in zip(tabs, names):
            with tab:
                st.plotly_chart(_index_map(name, result.indices[name]))
                st.plotly_chart(_histogram(name, result.indices[name]))
        with tabs[len(names)]:
            mask = sample.valid_mask.astype(int)
            fig = px.imshow(mask, color_continuous_scale=[[0, MISSING_PIXEL_GRAY], [1, "#2f8f3a"]],
                            zmin=0, zmax=1, aspect="equal")
            fig.update_layout(coloraxis_showscale=False, height=420, margin=dict(l=10, r=10, t=30, b=10),
                              title="Valid pixels (green) vs invalid / nodata pixels (grey)")
            fig.update_xaxes(showticklabels=False)
            fig.update_yaxes(showticklabels=False)
            st.plotly_chart(fig)
            st.write(f"{sample.n_total - sample.n_valid} invalid pixel(s) of {sample.n_total}.")
        with tabs[len(names) + 1]:
            rgb = _true_colour(sample)
            if rgb is None:
                st.info("Red, Green and Blue bands are needed for a true-colour preview.")
            else:
                st.image(rgb, caption="Contrast-stretched true-colour composite (display only)", width=420)
    else:
        st.subheader("Index distributions across observations")
        for name, array in result.indices.items():
            st.plotly_chart(_histogram(name, array))

    with st.expander("Index formulas and references"):
        for name, formula in INDEX_FORMULAS.items():
            st.markdown(f"**{INDEX_LABELS[name]}**: `{formula}`")
        st.markdown(
            "SAVI uses L = 0.5 (configurable). EVI uses MODIS coefficients and values outside "
            "[-1, 1] are treated as undefined. References: Rouse et al. 1974 (NDVI); Barnes et al. "
            "2000 (NDRE); Gitelson et al. 1996 (GNDVI); Huete 1988 (SAVI); Huete et al. 2002 (EVI)."
        )
