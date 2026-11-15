"""Turning per-pixel index maps into one fixed-order feature vector per sample.

The classifier never sees raw pixels. It sees these summary features, and the
same function is used when the demonstration model is trained and when it
makes predictions, so preprocessing is always consistent.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# The order matters: model artifacts store coefficients in exactly this order.
FEATURE_ORDER = [
    "ndvi_mean",
    "ndre_mean",
    "gndvi_mean",
    "savi_mean",
    "evi_mean",
    "ndvi_std",
    "ndvi_p10",
    "low_vigor_fraction",
    "ndre_ndvi_ratio",
]

FEATURE_DESCRIPTIONS = {
    "ndvi_mean": "Average NDVI - overall green biomass / canopy vigour",
    "ndre_mean": "Average NDRE - sensitive to canopy chlorophyll content",
    "gndvi_mean": "Average GNDVI - sensitive to chlorophyll concentration",
    "savi_mean": "Average SAVI - NDVI corrected for soil background",
    "evi_mean": "Average EVI - vigour index less prone to saturation",
    "ndvi_std": "Spatial standard deviation of NDVI - patchiness of the canopy",
    "ndvi_p10": "10th percentile of NDVI - how poor the weakest areas are",
    "low_vigor_fraction": "Fraction of valid pixels with NDVI below the low-vigour threshold",
    "ndre_ndvi_ratio": "NDRE / NDVI - chlorophyll signal relative to biomass signal",
}


@dataclass
class FeatureSet:
    values: dict[str, float]
    n_valid: int
    n_total: int
    warnings: list[str] = field(default_factory=list)

    @property
    def valid_fraction(self) -> float:
        return self.n_valid / self.n_total if self.n_total else 0.0

    @property
    def missing(self) -> list[str]:
        return [name for name in FEATURE_ORDER if not np.isfinite(self.values.get(name, np.nan))]

    def as_vector(self) -> np.ndarray:
        return np.array([self.values.get(name, np.nan) for name in FEATURE_ORDER], dtype="float64")


def _finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def extract_features(
    indices: dict[str, np.ndarray],
    valid_mask: np.ndarray,
    low_vigor_ndvi: float = 0.5,
    min_observations: int = 25,
) -> FeatureSet:
    """Summarise index maps into the FEATURE_ORDER features (NaN when unavailable)."""
    values = {name: float("nan") for name in FEATURE_ORDER}
    warnings = []

    for index_name in ("ndvi", "ndre", "gndvi", "savi", "evi"):
        if index_name in indices:
            finite = _finite(indices[index_name])
            if finite.size:
                values[f"{index_name}_mean"] = float(finite.mean())

    ndvi_values = _finite(indices["ndvi"]) if "ndvi" in indices else np.array([])
    if ndvi_values.size:
        values["ndvi_std"] = float(ndvi_values.std())
        values["ndvi_p10"] = float(np.percentile(ndvi_values, 10))
        values["low_vigor_fraction"] = float((ndvi_values < low_vigor_ndvi).mean())
        if ndvi_values.size < min_observations:
            warnings.append(
                f"Only {ndvi_values.size} valid observation(s); spatial features "
                "(ndvi_std, ndvi_p10, low_vigor_fraction) are unreliable below "
                f"{min_observations}."
            )

    ndvi_mean, ndre_mean = values["ndvi_mean"], values["ndre_mean"]
    if np.isfinite(ndvi_mean) and np.isfinite(ndre_mean) and ndvi_mean > 0.05:
        values["ndre_ndvi_ratio"] = ndre_mean / ndvi_mean

    return FeatureSet(values, int(valid_mask.sum()), int(valid_mask.size), warnings)


def summarize_indices(indices: dict[str, np.ndarray]) -> pd.DataFrame:
    """Descriptive statistics for each index map (for display and reports)."""
    rows = []
    for name, array in indices.items():
        finite = _finite(array)
        row = {"index": name.upper(), "valid_values": int(finite.size), "undefined_values": int(array.size - finite.size)}
        if finite.size:
            row.update(
                mean=float(finite.mean()),
                std=float(finite.std()),
                min=float(finite.min()),
                p10=float(np.percentile(finite, 10)),
                median=float(np.median(finite)),
                p90=float(np.percentile(finite, 90)),
                max=float(finite.max()),
            )
        rows.append(row)
    return pd.DataFrame(rows)
