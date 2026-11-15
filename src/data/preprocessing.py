"""Turning raw rasters or tables into clean, masked `SpectralSample` objects.

A SpectralSample is one unit of analysis (a field plot, a zone, or a group
of CSV rows sharing a sample_id). It holds reflectance arrays for each
available band and a boolean mask of valid observations.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.ingestion import RasterInput
from src.data.validation import (
    ValidationError,
    check_reflectance_scale,
    coerce_numeric_columns,
    resolve_csv_columns,
    validate_band_arrays,
    validate_band_mapping,
)


@dataclass
class SpectralSample:
    sample_id: str
    source_name: str
    kind: str  # "raster" or "points"
    bands: dict[str, np.ndarray]
    valid_mask: np.ndarray
    warnings: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def n_total(self) -> int:
        return int(self.valid_mask.size)

    @property
    def n_valid(self) -> int:
        return int(self.valid_mask.sum())

    @property
    def valid_fraction(self) -> float:
        return self.n_valid / self.n_total if self.n_total else 0.0

    @property
    def shape(self) -> tuple:
        return self.valid_mask.shape


def build_valid_mask(
    bands: dict[str, np.ndarray], nodata: float | None, min_valid: float, max_valid: float
) -> np.ndarray:
    """A pixel is valid only if every available band is finite, not nodata and in range."""
    first = next(iter(bands.values()))
    mask = np.ones(first.shape, dtype=bool)
    for array in bands.values():
        mask &= np.isfinite(array)
        if nodata is not None:
            mask &= array != nodata
        # Compare only finite values to avoid warnings; NaNs are already excluded.
        with np.errstate(invalid="ignore"):
            mask &= (array >= min_valid) & (array <= max_valid)
    return mask


def _describe_mask(mask: np.ndarray, warnings: list[str]) -> None:
    invalid = int(mask.size - mask.sum())
    if invalid:
        warnings.append(
            f"{invalid} of {mask.size} observations ({invalid / mask.size:.1%}) are invalid "
            "(nodata, missing, or outside the reflectance range) and were excluded."
        )


def _missing_band_warnings(bands: dict[str, np.ndarray], all_bands: list[str]) -> list[str]:
    missing = [band for band in all_bands if band not in bands]
    if not missing:
        return []
    return [
        "Band(s) not provided: " + ", ".join(missing)
        + ". Indices that need them cannot be calculated."
    ]


def sample_from_raster(
    raster: RasterInput,
    band_mapping: dict[str, int | None],
    settings: dict,
    scale_factor: float = 1.0,
    sample_id: str | None = None,
) -> SpectralSample:
    """Select and scale the mapped bands of a multiband raster."""
    mapping = validate_band_mapping(band_mapping, raster.band_count)
    if scale_factor <= 0:
        raise ValidationError("The reflectance scale factor must be a positive number.")

    raw = {name: raster.array[number - 1] for name, number in mapping.items()}
    return _sample_from_band_arrays(
        raw, raster.nodata, settings, scale_factor,
        sample_id or raster.source_name, raster.source_name, "raster",
        {**raster.metadata, "reader": raster.reader, "band_mapping": mapping,
         "nodata": raster.nodata, "scale_factor": scale_factor},
    )


def sample_from_band_files(
    rasters: dict[str, RasterInput], settings: dict, scale_factor: float = 1.0
) -> SpectralSample:
    """Combine separate single-band rasters (one file per spectral band)."""
    raw = {}
    nodata_values = set()
    for band_name, raster in rasters.items():
        if raster.band_count != 1:
            raise ValidationError(
                f"'{raster.source_name}' (uploaded as {band_name}) has {raster.band_count} "
                "bands; single-band files are expected here."
            )
        raw[band_name] = raster.array[0]
        if raster.nodata is not None:
            nodata_values.add(raster.nodata)
    if len(nodata_values) > 1:
        raise ValidationError(f"The band files use different nodata values: {nodata_values}.")

    names = ", ".join(r.source_name for r in rasters.values())
    return _sample_from_band_arrays(
        raw, nodata_values.pop() if nodata_values else None, settings, scale_factor,
        "band-files", names, "raster", {"scale_factor": scale_factor},
    )


def _sample_from_band_arrays(
    raw, nodata, settings, scale_factor, sample_id, source, kind, metadata, allow_empty=False
):
    validate_band_arrays(raw)
    refl = settings["reflectance"]

    # Mask nodata before scaling so the nodata value is matched exactly.
    nodata_mask = {name: (array == nodata) if nodata is not None else None for name, array in raw.items()}
    bands = {}
    for name, array in raw.items():
        scaled = array / scale_factor
        if nodata_mask[name] is not None:
            scaled = np.where(nodata_mask[name], np.nan, scaled)
        bands[name] = scaled

    check_reflectance_scale(bands, refl["dn_detection_threshold"])
    mask = build_valid_mask(bands, None, refl["min_valid"], refl["max_valid"])
    if mask.sum() == 0 and not allow_empty:
        raise ValidationError("No valid observations remain after masking nodata and invalid values.")

    warnings = _missing_band_warnings(bands, settings["bands"])
    _describe_mask(mask, warnings)
    return SpectralSample(sample_id, source, kind, bands, mask, warnings, metadata)


def samples_from_table(
    frame: pd.DataFrame, source_name: str, settings: dict, scale_factor: float = 1.0
) -> list[SpectralSample]:
    """Split a CSV of band reflectances into one SpectralSample per sample_id.

    Without a sample_id column, all rows form a single sample.
    """
    columns = resolve_csv_columns(frame, settings["csv_aliases"])
    frame = coerce_numeric_columns(frame, columns)

    if "sample_id" in columns:
        id_column = columns["sample_id"]
        frame = frame.assign(_sample=frame[id_column].astype(str))
    else:
        frame = frame.assign(_sample="all-rows")

    band_columns = {name: col for name, col in columns.items() if name != "sample_id"}
    samples = []
    for sample_id, group in frame.groupby("_sample", sort=False):
        raw = {name: group[col].to_numpy(dtype="float64") for name, col in band_columns.items()}
        sample = _sample_from_band_arrays(
            raw, None, settings, scale_factor, str(sample_id), source_name, "points",
            {"rows": len(group), "scale_factor": scale_factor, "columns": band_columns},
            allow_empty=True,  # one bad sample should not reject the whole file
        )
        # Keep demonstration ground-truth labels if the file carries them.
        for extra in ("label", "synthetic_label", "is_synthetic", "crop"):
            if extra in group.columns:
                sample.metadata[extra] = str(group[extra].iloc[0])
        samples.append(sample)
    return samples
