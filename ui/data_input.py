import json

import pandas as pd
import streamlit as st

from src.config import DEMO_DATA_DIR
from src.data.ingestion import RASTERIO_AVAILABLE, read_csv_table, read_raster
from src.data.preprocessing import sample_from_band_files, sample_from_raster, samples_from_table
from src.data.synthetic import BANDS
from src.data.validation import ValidationError
from ui.common import get_settings, set_samples

BAND_LABELS = {"blue": "Blue", "green": "Green", "red": "Red", "red_edge": "Red Edge", "nir": "NIR"}

CSV_TEMPLATE = (
    "sample_id,crop,blue,green,red,red_edge,nir\n"
    "plotA,wheat,0.035,0.085,0.042,0.29,0.48\n"
    "plotA,wheat,0.036,0.088,0.045,0.30,0.47\n"
)


def _load_manifest() -> list[dict]:
    path = DEMO_DATA_DIR / "manifest.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))["files"]


def load_demo_file(name: str) -> list:
    """Load a bundled demo entry (name as listed in manifest.json) into SpectralSamples."""
    settings = get_settings()
    demo_mapping = next(iter(settings["band_presets"].values()))
    if name.startswith("bands/"):
        rasters = {b: read_raster(DEMO_DATA_DIR / "bands" / f"water_stress_{b}.tif") for b in BANDS}
        return [sample_from_band_files(rasters, settings)]
    if name.endswith(".csv"):
        frame, source = read_csv_table(DEMO_DATA_DIR / name)
        return samples_from_table(frame, source, settings)
    path = DEMO_DATA_DIR / name
    if path.parent != DEMO_DATA_DIR or not path.exists():
        raise ValidationError(f"Unknown demo file '{name}'.")
    scale = 10000.0 if "x10000" in name else 1.0
    return [sample_from_raster(read_raster(path), demo_mapping, settings, scale, path.stem)]


def _show_loaded(samples: list) -> None:
    rows = [{
        "sample_id": s.sample_id,
        "type": s.kind,
        "shape": " x ".join(map(str, s.shape)),
        "bands": ", ".join(s.bands),
        "valid observations": f"{s.n_valid}/{s.n_total} ({s.valid_fraction:.0%})",
        "warnings": len(s.warnings),
    } for s in samples]
    st.success(f"Loaded {len(samples)} sample(s). Continue to **Vegetation Indices** or **Crop-Stress Assessment**.")
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    for sample in samples:
        for warning in sample.warnings:
            st.warning(f"{sample.sample_id}: {warning}")


def _band_mapping_inputs(band_count: int, key: str) -> dict[str, int | None]:
    presets = get_settings()["band_presets"]
    preset_name = st.selectbox("Band-order preset", list(presets) + ["Custom"], key=f"{key}_preset")
    defaults = presets.get(preset_name, {})
    st.caption(
        "Raster band number (1-based) holding each spectral band. Use 0 if your sensor "
        "does not provide that band. Always check your sensor/export documentation."
    )
    columns = st.columns(5)
    mapping = {}
    for column, band in zip(columns, BANDS):
        default = defaults.get(band, 0)
        default = default if default <= band_count else 0
        mapping[band] = column.number_input(
            BAND_LABELS[band], min_value=0, max_value=max(band_count, 1), value=default,
            step=1, key=f"{key}_{band}_{preset_name}",
        ) or None
    return mapping


def _scale_input(key: str) -> float:
    return st.number_input(
        "Reflectance scale factor (stored value ÷ factor = reflectance)", min_value=0.0001,
        value=1.0, step=1.0, key=key,
        help="1 for float reflectance (0-1). 10000 for Sentinel-2 L2A or other x10000 products.",
    )


def _demo_tab() -> None:
    settings = get_settings()
    manifest = _load_manifest()
    if not manifest:
        st.error("Demo data not found. Run `python scripts/generate_demo_data.py` to recreate it.")
        return
    st.write("All bundled files are **synthetic** (see `data/demo/README.md`). Pick one:")
    usable = [m for m in manifest if not m["file"].startswith("invalid")]
    choice = st.selectbox(
        "Demonstration file", usable, format_func=lambda m: m["file"], key="demo_choice"
    )
    st.markdown(f"*{choice['description']}*  \nExpected behaviour: **{choice['expected_outcome']}**")

    if st.button("Load selected demo file", type="primary"):
        try:
            if "x10000" in choice["file"]:
                st.info("This file stores reflectance x 10000, so the scale factor 10000 is applied.")
            samples = load_demo_file(choice["file"])
            set_samples(samples, f"Bundled demo: {choice['file']}")
            _show_loaded(samples)
        except ValidationError as error:
            st.error(f"Validation failed: {error}")

    with st.expander("Try a deliberately broken file (shows validation messages)"):
        broken = sorted(p.name for p in (DEMO_DATA_DIR / "invalid").glob("*"))
        name = st.selectbox("Broken file", broken, key="broken_choice")
        if st.button("Validate broken file"):
            try:
                path = DEMO_DATA_DIR / "invalid" / name
                if name.endswith(".csv"):
                    frame, source = read_csv_table(path)
                    samples_from_table(frame, source, settings)
                else:
                    read_raster(path)
                st.success("Unexpectedly accepted.")
            except ValidationError as error:
                st.error(f"Validation failed (as expected): {error}")


def _multiband_tab() -> None:
    settings = get_settings()
    uploaded = st.file_uploader("Multiband GeoTIFF / TIFF", type=["tif", "tiff"], key="mb_upload")
    if not uploaded:
        return
    try:
        raster = read_raster(uploaded)
    except ValidationError as error:
        st.error(str(error))
        return
    st.write(
        f"Read **{raster.source_name}** with `{raster.reader}`: {raster.band_count} band(s), "
        f"{raster.array.shape[1]} x {raster.array.shape[2]} pixels, nodata = {raster.nodata}, "
        f"georeferenced: {raster.metadata.get('georeferenced')}"
    )
    mapping = _band_mapping_inputs(raster.band_count, "mb")
    scale = _scale_input("mb_scale")
    if st.button("Validate and load", type="primary", key="mb_go"):
        try:
            sample = sample_from_raster(raster, mapping, settings, scale)
            set_samples([sample], f"Upload: {raster.source_name}")
            _show_loaded([sample])
        except ValidationError as error:
            st.error(f"Validation failed: {error}")


def _band_files_tab() -> None:
    settings = get_settings()
    st.caption("Upload one single-band raster per spectral band. All must share the same grid. Red and NIR are required.")
    columns = st.columns(5)
    uploads = {
        band: column.file_uploader(BAND_LABELS[band], type=["tif", "tiff"], key=f"bf_{band}")
        for column, band in zip(columns, BANDS)
    }
    scale = _scale_input("bf_scale")
    if st.button("Validate and load", type="primary", key="bf_go"):
        provided = {band: f for band, f in uploads.items() if f is not None}
        try:
            missing = [BAND_LABELS[b] for b in ("red", "nir") if b not in provided]
            if missing:
                raise ValidationError("Required band file(s) missing: " + ", ".join(missing))
            rasters = {band: read_raster(f) for band, f in provided.items()}
            sample = sample_from_band_files(rasters, settings, scale)
            set_samples([sample], "Upload: separate band files")
            _show_loaded([sample])
        except ValidationError as error:
            st.error(f"Validation failed: {error}")


def _csv_tab() -> None:
    settings = get_settings()
    aliases = settings["csv_aliases"]
    st.markdown(
        "One row per observation (pixel or point). Rows with the same `sample_id` are analysed "
        "together as one plot; without `sample_id` all rows form one sample. Optional columns: "
        "`crop`, and `label` for evaluating on your own labelled data."
    )
    st.dataframe(
        pd.DataFrame([{"standard name": k, "accepted column names": ", ".join(v)} for k, v in aliases.items()]),
        hide_index=True,
    )
    st.download_button("Download CSV template", CSV_TEMPLATE, "crop_stress_template.csv", "text/csv")
    uploaded = st.file_uploader("CSV table", type=["csv"], key="csv_upload")
    scale = _scale_input("csv_scale")
    if uploaded and st.button("Validate and load", type="primary", key="csv_go"):
        try:
            frame, source = read_csv_table(uploaded)
            samples = samples_from_table(frame, source, settings, scale)
            set_samples(samples, f"Upload: {source}")
            _show_loaded(samples)
        except ValidationError as error:
            st.error(f"Validation failed: {error}")


def render() -> None:
    st.title("Data Input")
    st.markdown(
        "**Input conventions.** Values must be **surface reflectance** (0-1) after applying the scale "
        "factor - not raw digital numbers, and not radiance. Radiometric calibration (e.g. with a "
        "reflectance panel) and band co-registration must be done beforehand. Pixels that are "
        "nodata, non-finite or outside 0-1.2 are masked as invalid."
    )
    if not RASTERIO_AVAILABLE:
        st.caption(
            "Raster reader: `tifffile` (lightweight). Install the optional `rasterio` package for "
            "broader GeoTIFF support and CRS reading - see README."
        )
    if st.session_state.input_label:
        st.info(f"Currently loaded: **{st.session_state.input_label}** ({len(st.session_state.samples)} sample(s))")

    tabs = st.tabs(["Bundled demo data", "Multiband GeoTIFF", "Separate band files", "CSV table"])
    with tabs[0]:
        _demo_tab()
    with tabs[1]:
        _multiband_tab()
    with tabs[2]:
        _band_files_tab()
    with tabs[3]:
        _csv_tab()
