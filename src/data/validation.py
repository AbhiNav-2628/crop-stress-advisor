"""Input validation. Every check raises ValidationError with a message that a
user (not only a developer) can act on."""

import numpy as np
import pandas as pd

REQUIRED_BANDS = ("red", "nir")  # the minimum needed for NDVI


class ValidationError(ValueError):
    """Raised when input data cannot be processed safely."""


def validate_band_mapping(mapping: dict[str, int | None], band_count: int) -> dict[str, int]:
    """Check a {band_name: 1-based band number} mapping against a raster.

    Bands mapped to None/0 are treated as "not provided by this sensor".
    Returns the cleaned mapping containing only the provided bands.
    """
    cleaned = {name: int(number) for name, number in mapping.items() if number}

    missing_required = [band for band in REQUIRED_BANDS if band not in cleaned]
    if missing_required:
        raise ValidationError(
            "The band mapping must include at least Red and NIR. Missing: "
            + ", ".join(missing_required)
        )

    for name, number in cleaned.items():
        if not 1 <= number <= band_count:
            raise ValidationError(
                f"Band '{name}' is mapped to band {number}, but the raster only has "
                f"{band_count} band(s)."
            )

    used = list(cleaned.values())
    duplicates = sorted({n for n in used if used.count(n) > 1})
    if duplicates:
        raise ValidationError(
            f"Raster band(s) {duplicates} are assigned to more than one spectral band. "
            "Each spectral band needs its own raster band."
        )
    return cleaned


def validate_band_arrays(bands: dict[str, np.ndarray]) -> None:
    """All bands must be 2-D (rasters) or 1-D (point tables) and share one shape."""
    missing_required = [band for band in REQUIRED_BANDS if band not in bands]
    if missing_required:
        raise ValidationError("Missing required band(s): " + ", ".join(missing_required))

    shapes = {name: array.shape for name, array in bands.items()}
    first_shape = next(iter(shapes.values()))
    if any(len(shape) not in (1, 2) for shape in shapes.values()):
        raise ValidationError(f"Band arrays must be 1-D or 2-D, got shapes {shapes}.")
    if any(shape != first_shape for shape in shapes.values()):
        raise ValidationError(
            "Bands are not aligned: they have different dimensions "
            f"{shapes}. Resample/co-register the rasters to one grid first."
        )
    if min(first_shape) == 0:
        raise ValidationError("The input contains no pixels or rows.")


def check_reflectance_scale(bands: dict[str, np.ndarray], dn_threshold: float) -> None:
    """Reject data that looks like raw digital numbers instead of reflectance."""
    for name, array in bands.items():
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            continue
        high = float(np.percentile(finite, 99))
        if high > dn_threshold:
            raise ValidationError(
                f"Band '{name}' has values up to {high:.1f}, which looks like digital numbers "
                "or scaled integers rather than surface reflectance (0-1). Set the "
                "reflectance scale factor (for example 10000 for Sentinel-2 L2A, or the "
                "factor given by your sensor's processing software)."
            )


def resolve_csv_columns(frame: pd.DataFrame, aliases: dict[str, list[str]]) -> dict[str, str]:
    """Map standard names (blue, green, ..., sample_id) to the CSV's actual columns."""
    lookup = {str(column).strip().lower(): column for column in frame.columns}
    resolved = {}
    for standard_name, names in aliases.items():
        for alias in names:
            if alias.lower() in lookup:
                resolved[standard_name] = lookup[alias.lower()]
                break

    missing_required = [band for band in REQUIRED_BANDS if band not in resolved]
    if missing_required:
        expected = {band: aliases[band] for band in missing_required}
        raise ValidationError(
            f"The CSV is missing required column(s) {missing_required}. "
            f"Accepted names: {expected}. Found columns: {list(frame.columns)}."
        )
    return resolved


def coerce_numeric_columns(frame: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    """Convert band columns to numbers. Fails if a column is entirely non-numeric."""
    converted = frame.copy()
    for standard_name, column in columns.items():
        if standard_name == "sample_id":
            continue
        numeric = pd.to_numeric(converted[column], errors="coerce")
        if numeric.notna().sum() == 0:
            raise ValidationError(
                f"Column '{column}' ({standard_name}) contains no numeric values."
            )
        converted[column] = numeric
    return converted
