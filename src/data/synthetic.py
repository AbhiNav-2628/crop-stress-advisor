"""Deterministic SYNTHETIC multispectral plots for demonstration and testing.

Nothing produced here is a field observation. Each synthetic plot is built
with a simple linear mixing model:

    pixel = canopy_fraction x leaf_spectrum + (1 - canopy_fraction) x soil_spectrum

where the leaf spectrum is a blend of a healthy leaf and a "stressed" leaf.
Each scenario changes the canopy fraction, the stressed-leaf type and the
spatial pattern (uniform vs. patchy) in a way loosely inspired by textbook
descriptions of plant stress responses. The endmember spectra are plausible
illustrative values, not measurements from a spectral library.

Because severity is drawn from a range that includes mild cases, scenarios
overlap on purpose, so a classifier fitted to this data is not perfect.
"""

import numpy as np

BANDS = ["blue", "green", "red", "red_edge", "nir"]

# Illustrative surface-reflectance endmembers: [blue, green, red, red_edge, nir]
ENDMEMBERS = {
    "healthy_leaf": np.array([0.035, 0.085, 0.040, 0.300, 0.500]),
    "soil": np.array([0.090, 0.130, 0.170, 0.210, 0.260]),
    "chlorotic_leaf": np.array([0.045, 0.120, 0.075, 0.360, 0.480]),  # low chlorophyll
    "wilted_leaf": np.array([0.045, 0.095, 0.060, 0.280, 0.400]),     # water deficit
    "necrotic_leaf": np.array([0.060, 0.100, 0.120, 0.250, 0.330]),   # lesions / dead tissue
}

SCENARIOS = ["healthy", "water_stress", "nutrient_stress", "pest_stress", "disease_stress"]
NODATA_VALUE = -9999.0


def smooth_field(size: int, cells: int, rng: np.random.Generator) -> np.ndarray:
    """Spatially smooth random field in [0, 1] made by bilinear upsampling of coarse noise."""
    coarse = rng.random((cells + 1, cells + 1))
    coarse_axis = np.linspace(0, 1, cells + 1)
    fine_axis = np.linspace(0, 1, size)
    rows = np.array([np.interp(fine_axis, coarse_axis, row) for row in coarse])        # (cells+1, size)
    field = np.array([np.interp(fine_axis, coarse_axis, col) for col in rows.T]).T     # (size, size)
    field -= field.min()
    return field / field.max() if field.max() > 0 else field


def patch_mask(size: int, area_fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Boolean mask covering roughly `area_fraction` of the plot in clumped patches."""
    field = smooth_field(size, cells=6, rng=rng)
    return field >= np.quantile(field, 1 - area_fraction)


def _pick(fixed: float | None, rng: np.random.Generator, low: float, high: float) -> float:
    return fixed if fixed is not None else rng.uniform(low, high)


def synthesize_plot(
    scenario: str,
    rng: np.random.Generator,
    size: int = 64,
    severity: float | None = None,
    typical: bool = False,
) -> np.ndarray:
    """Return a (5, size, size) reflectance cube for one synthetic plot.

    `severity` (0-1) fixes the stress intensity; by default it is drawn at random
    from a range that includes mild, hard-to-detect cases.
    `typical=True` fixes the nuisance factors (leaf spectrum, soil brightness,
    canopy cover) at central values. The bundled demo files use this so that each
    one is a readable example of its scenario; training data never does.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario}'. Choose from {SCENARIOS}.")

    # Nuisance variation that has nothing to do with stress: crop variety and
    # growth stage change the leaf spectrum, soils differ in brightness, and
    # canopy cover varies between plots. This keeps the task from being trivial.
    if typical:
        healthy, soil = ENDMEMBERS["healthy_leaf"], ENDMEMBERS["soil"]
        cover_amplitude, base_cover = 0.08, 0.85
    else:
        healthy = ENDMEMBERS["healthy_leaf"] * rng.normal(1.0, 0.05, 5)
        soil = ENDMEMBERS["soil"] * rng.uniform(0.75, 1.3)
        cover_amplitude, base_cover = rng.uniform(0.05, 0.14), rng.uniform(0.74, 0.93)
    cover_noise = (smooth_field(size, cells=8, rng=rng) - 0.5) * cover_amplitude
    canopy = np.full((size, size), base_cover) + cover_noise
    stress_weight = np.zeros((size, size))  # 0 = healthy leaf, 1 = fully stressed leaf
    stressed_leaf = healthy

    if scenario == "water_stress":
        severity = _pick(severity, rng, 0.2, 0.9)
        stressed_leaf = ENDMEMBERS["wilted_leaf"]
        stress_weight[:] = severity                        # uniform across the field
        canopy -= 0.20 * severity                          # leaf rolling reduces cover
    elif scenario == "nutrient_stress":
        severity = _pick(severity, rng, 0.25, 0.95)
        stressed_leaf = ENDMEMBERS["chlorotic_leaf"]
        gradient = np.linspace(0.75, 1.0, size)[np.newaxis, :]  # gentle gradient, e.g. uneven fertilisation
        if rng.random() < 0.5:
            gradient = gradient.T
        stress_weight[:] = severity * gradient
        canopy -= 0.06 * severity
    elif scenario == "pest_stress":
        severity = _pick(severity, rng, 0.25, 1.0)
        patches = patch_mask(size, rng.uniform(0.08, 0.35), rng)
        canopy[patches] -= 0.70 * severity                 # defoliation exposes soil
        stressed_leaf = ENDMEMBERS["wilted_leaf"]
        stress_weight[patches] = 0.3 * severity
    elif scenario == "disease_stress":
        severity = _pick(severity, rng, 0.25, 1.0)
        patches = patch_mask(size, rng.uniform(0.10, 0.40), rng)
        stressed_leaf = 0.5 * ENDMEMBERS["necrotic_leaf"] + 0.5 * ENDMEMBERS["chlorotic_leaf"]
        stress_weight[patches] = severity                  # lesions concentrated in foci
        canopy[patches] -= 0.15 * severity

    canopy = np.clip(canopy, 0.05, 0.98)
    leaf = healthy[:, None, None] * (1 - stress_weight) + stressed_leaf[:, None, None] * stress_weight
    cube = canopy * leaf + (1 - canopy) * soil[:, None, None]

    cube *= 1.0 if typical else rng.uniform(0.93, 1.07)                        # illumination / calibration differences
    cube += rng.normal(0.0, 0.006, cube.shape)             # sensor noise
    return np.clip(cube, 0.001, 1.0)


def add_cloud_and_nodata(cube: np.ndarray, rng: np.random.Generator, fraction: float = 0.3) -> np.ndarray:
    """Mark a clumped region as nodata (e.g. masked cloud shadow or image edge)."""
    cube = cube.copy()
    mask = patch_mask(cube.shape[1], fraction, rng)
    cube[:, mask] = NODATA_VALUE
    return cube


def generate_dataset(n_per_class: int, seed: int, size: int = 64):
    """Yield (scenario, cube) pairs in a reproducible order."""
    rng = np.random.default_rng(seed)
    for scenario in SCENARIOS:
        for _ in range(n_per_class):
            yield scenario, synthesize_plot(scenario, rng, size)
