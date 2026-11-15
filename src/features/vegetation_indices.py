"""Vegetation-index formulas.

All functions take reflectance arrays (0-1) and return float arrays where any
undefined value (zero denominator, invalid input) is NaN instead of +/-inf.

References
----------
NDVI  - Rouse et al. (1974), NASA SP-351.
NDRE  - Barnes et al. (2000), Proc. 5th Int. Conf. on Precision Agriculture.
GNDVI - Gitelson, Kaufman & Merzlyak (1996), Remote Sensing of Environment 58.
SAVI  - Huete (1988), Remote Sensing of Environment 25.
EVI   - Huete et al. (2002), Remote Sensing of Environment 83.
"""

import numpy as np

# Which bands each index needs. Used to decide what can be computed.
INDEX_REQUIREMENTS = {
    "ndvi": ("nir", "red"),
    "ndre": ("nir", "red_edge"),
    "gndvi": ("nir", "green"),
    "savi": ("nir", "red"),
    "evi": ("nir", "red", "blue"),
}

INDEX_LABELS = {
    "ndvi": "NDVI - Normalized Difference Vegetation Index",
    "ndre": "NDRE - Normalized Difference Red Edge index",
    "gndvi": "GNDVI - Green Normalized Difference Vegetation Index",
    "savi": "SAVI - Soil Adjusted Vegetation Index",
    "evi": "EVI - Enhanced Vegetation Index",
}

INDEX_FORMULAS = {
    "ndvi": "(NIR - Red) / (NIR + Red)",
    "ndre": "(NIR - RedEdge) / (NIR + RedEdge)",
    "gndvi": "(NIR - Green) / (NIR + Green)",
    "savi": "((NIR - Red) / (NIR + Red + L)) x (1 + L)",
    "evi": "2.5 x (NIR - Red) / (NIR + 6 x Red - 7.5 x Blue + 1)",
}

DEFAULT_MIN_DENOMINATOR = 1e-6


def safe_divide(numerator, denominator, min_denominator=DEFAULT_MIN_DENOMINATOR) -> np.ndarray:
    """Element-wise division that yields NaN where the denominator is ~0 or not finite."""
    numerator = np.asarray(numerator, dtype="float64")
    denominator = np.asarray(denominator, dtype="float64")
    usable = np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denominator) >= min_denominator)
    result = np.full(np.broadcast(numerator, denominator).shape, np.nan)
    np.divide(numerator, denominator, out=result, where=usable)
    return result


def normalized_difference(a, b, min_denominator=DEFAULT_MIN_DENOMINATOR) -> np.ndarray:
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    return safe_divide(a - b, a + b, min_denominator)


def ndvi(nir, red, min_denominator=DEFAULT_MIN_DENOMINATOR):
    return normalized_difference(nir, red, min_denominator)


def ndre(nir, red_edge, min_denominator=DEFAULT_MIN_DENOMINATOR):
    return normalized_difference(nir, red_edge, min_denominator)


def gndvi(nir, green, min_denominator=DEFAULT_MIN_DENOMINATOR):
    return normalized_difference(nir, green, min_denominator)


def savi(nir, red, soil_factor=0.5, min_denominator=DEFAULT_MIN_DENOMINATOR):
    nir = np.asarray(nir, dtype="float64")
    red = np.asarray(red, dtype="float64")
    return safe_divide(nir - red, nir + red + soil_factor, min_denominator) * (1 + soil_factor)


def evi(nir, red, blue, min_denominator=DEFAULT_MIN_DENOMINATOR):
    """EVI with the MODIS coefficients (G=2.5, C1=6, C2=7.5, L=1).

    EVI is only meaningful for reflectance data. Values outside [-1, 1]
    usually come from noisy or atmospherically uncorrected pixels (e.g. very
    high blue reflectance), so they are set to NaN rather than kept.
    """
    nir = np.asarray(nir, dtype="float64")
    red = np.asarray(red, dtype="float64")
    blue = np.asarray(blue, dtype="float64")
    value = 2.5 * safe_divide(nir - red, nir + 6.0 * red - 7.5 * blue + 1.0, min_denominator)
    with np.errstate(invalid="ignore"):
        value[(value < -1.0) | (value > 1.0)] = np.nan
    return value


def compute_indices(
    bands: dict[str, np.ndarray],
    valid_mask: np.ndarray,
    soil_factor: float = 0.5,
    min_denominator: float = DEFAULT_MIN_DENOMINATOR,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Compute every index whose bands are available.

    Returns (indices, skipped) where `skipped` maps index name -> reason.
    Invalid pixels (valid_mask == False) are NaN in every index.
    """
    masked = {name: np.where(valid_mask, array, np.nan) for name, array in bands.items()}
    indices, skipped = {}, {}

    for name, required in INDEX_REQUIREMENTS.items():
        missing = [band for band in required if band not in masked]
        if missing:
            skipped[name] = f"requires band(s) not provided: {', '.join(missing)}"
            continue
        if name == "ndvi":
            indices[name] = ndvi(masked["nir"], masked["red"], min_denominator)
        elif name == "ndre":
            indices[name] = ndre(masked["nir"], masked["red_edge"], min_denominator)
        elif name == "gndvi":
            indices[name] = gndvi(masked["nir"], masked["green"], min_denominator)
        elif name == "savi":
            indices[name] = savi(masked["nir"], masked["red"], soil_factor, min_denominator)
        elif name == "evi":
            indices[name] = evi(masked["nir"], masked["red"], masked["blue"], min_denominator)
    return indices, skipped
