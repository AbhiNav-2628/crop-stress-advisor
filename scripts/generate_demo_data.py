"""Generate the bundled SYNTHETIC demonstration files in data/demo/.

Run from the project root:
    python scripts/generate_demo_data.py

The output is fully determined by DEMO_SEED, so re-running it reproduces the
same files. None of these files are field observations.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEMO_DATA_DIR  # noqa: E402
from src.data.synthetic import (  # noqa: E402
    BANDS, NODATA_VALUE, SCENARIOS, add_cloud_and_nodata, synthesize_plot,
)

DEMO_SEED = 2025
# Demo plots use a fixed, clearly expressed stress severity and central nuisance
# values ("typical=True") so each file is a readable example of its scenario.
# Training/evaluation data uses random severities and nuisance factors.
DEMO_SEVERITY = 0.8
PIXEL_SIZE_M = 0.05                 # 5 cm, typical of a low-altitude UAV flight
ORIGIN_UTM = (500000.0, 1900000.0)  # arbitrary location in UTM zone 43N (synthetic)
EPSG_UTM_43N = 32643


def geotiff_tags(nodata: float) -> list[tuple]:
    """Minimal GeoTIFF + GDAL nodata tags so GIS software can place the image."""
    geokeys = [1, 1, 0, 3,            # header: version, revision, minor, number of keys
               1024, 0, 1, 1,         # GTModelType = projected
               1025, 0, 1, 1,         # GTRasterType = PixelIsArea
               3072, 0, 1, EPSG_UTM_43N]
    return [
        (33550, "d", 3, (PIXEL_SIZE_M, PIXEL_SIZE_M, 0.0), True),            # ModelPixelScale
        (33922, "d", 6, (0.0, 0.0, 0.0, ORIGIN_UTM[0], ORIGIN_UTM[1], 0.0), True),  # ModelTiepoint
        (34735, "H", len(geokeys), geokeys, True),                            # GeoKeyDirectory
        (42113, "s", 0, f"{nodata:g}", True),                                 # GDAL_NODATA
    ]


def write_geotiff(path: Path, cube: np.ndarray, nodata: float, dtype="float32") -> None:
    tifffile.imwrite(
        path, cube.astype(dtype), photometric="minisblack", planarconfig="separate",
        extratags=geotiff_tags(nodata), compression="zlib",
    )


def sample_points(cube: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Pick n random pixels, as if spectra were exported from a GIS at sample points."""
    rows = rng.integers(0, cube.shape[1], n)
    cols = rng.integers(0, cube.shape[2], n)
    return cube[:, rows, cols].T  # (n, bands)


def main() -> None:
    rng = np.random.default_rng(DEMO_SEED)
    out = DEMO_DATA_DIR
    (out / "bands").mkdir(parents=True, exist_ok=True)
    (out / "invalid").mkdir(parents=True, exist_ok=True)
    manifest = {"seed": DEMO_SEED, "synthetic": True, "files": []}

    def record(name, description, expected):
        manifest["files"].append({"file": name, "description": description, "expected_outcome": expected})

    # 1. One multiband GeoTIFF per scenario.
    cubes = {}
    for scenario in SCENARIOS:
        cube = synthesize_plot(scenario, rng, severity=DEMO_SEVERITY, typical=True)
        cubes[scenario] = cube
        name = f"plot_{scenario}.tif"
        write_geotiff(out / name, cube, NODATA_VALUE)
        record(name, f"Synthetic 64x64 plot, scenario '{scenario}', bands B,G,R,RE,NIR, float reflectance",
               scenario)

    # 1b. A mild case on purpose: the system should be allowed to be unsure.
    write_geotiff(out / "plot_mild_ambiguous.tif", synthesize_plot("water_stress", rng, severity=0.15, typical=True),
                  NODATA_VALUE)
    record("plot_mild_ambiguous.tif", "Very mild synthetic water-stress plot (severity 0.15)",
           "may be inconclusive or confused with healthy - shows uncertainty handling")

    # 2. Mostly nodata (e.g. cloud/shadow masked) -> must be inconclusive.
    cloudy = add_cloud_and_nodata(synthesize_plot("healthy", rng, typical=True), rng, fraction=0.6)
    write_geotiff(out / "plot_cloud_masked.tif", cloudy, NODATA_VALUE)
    record("plot_cloud_masked.tif", "Healthy synthetic plot with ~60% of pixels set to nodata",
           "inconclusive (too few valid pixels)")

    # 3. Bare soil / non-crop surface -> unsupported.
    soil = np.array([0.09, 0.13, 0.17, 0.21, 0.26])[:, None, None] * rng.uniform(0.9, 1.1, (1, 64, 64))
    soil = soil + rng.normal(0, 0.005, (5, 64, 64))
    write_geotiff(out / "plot_bare_soil.tif", np.clip(soil, 0.001, 1), NODATA_VALUE)
    record("plot_bare_soil.tif", "Synthetic bare-soil surface", "inconclusive (non-crop surface)")

    # 4. Same nutrient scenario stored as scaled 16-bit integers (reflectance x 10000).
    scaled = np.round(cubes["nutrient_stress"] * 10000)
    write_geotiff(out / "plot_nutrient_stress_uint16_x10000.tif", scaled, 0, dtype="uint16")
    record("plot_nutrient_stress_uint16_x10000.tif",
           "Nutrient scenario stored as uint16 reflectance x 10000 (nodata 0)",
           "validation error until the scale factor is set to 10000, then nutrient_stress")

    # 5. Same water-stress plot written as separate single-band files.
    for band_index, band in enumerate(BANDS):
        write_geotiff(out / "bands" / f"water_stress_{band}.tif", cubes["water_stress"][band_index:band_index + 1],
                      NODATA_VALUE)
    record("bands/water_stress_<band>.tif", "Water-stress plot as five single-band GeoTIFFs",
           "water_stress (use the 'Separate band files' input)")

    # 6. Point table: 80 sampled pixels per synthetic plot.
    crops = ["wheat", "rice", "maize", "cotton", "generic"]
    frames = []
    for i, scenario in enumerate(SCENARIOS):
        points = sample_points(synthesize_plot(scenario, rng, severity=DEMO_SEVERITY, typical=True), 80, rng)
        frame = pd.DataFrame(np.round(points, 5), columns=BANDS)
        frame.insert(0, "sample_id", f"P{i + 1:02d}-{scenario}")
        frame.insert(1, "crop", crops[i])
        frame["is_synthetic"] = True
        frame["synthetic_label"] = scenario
        frames.append(frame)
    points = pd.concat(frames, ignore_index=True)
    points.to_csv(out / "demo_points.csv", index=False)
    record("demo_points.csv", "80 synthetic pixel spectra per plot for 5 plots, grouped by sample_id",
           "one assessment per sample_id")

    # 7. The same table without the red-edge band.
    points.drop(columns=["red_edge"]).to_csv(out / "demo_points_no_red_edge.csv", index=False)
    record("demo_points_no_red_edge.csv", "demo_points.csv without the red_edge column",
           "NDVI/GNDVI/SAVI/EVI calculated; NDRE skipped; classification inconclusive")

    # 8. Deliberately broken inputs for demonstrating validation messages.
    (out / "invalid" / "missing_nir_column.csv").write_text(
        "sample_id,blue,green,red,red_edge\nA,0.03,0.08,0.04,0.3\n", encoding="utf-8")
    (out / "invalid" / "text_in_numeric_column.csv").write_text(
        "sample_id,blue,green,red,red_edge,nir\nA,x,y,z,w,v\nA,a,b,c,d,e\n", encoding="utf-8")
    (out / "invalid" / "not_a_tiff.tif").write_text("this is not an image", encoding="utf-8")
    record("invalid/*", "Malformed inputs", "clear validation error")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(manifest['files'])} demo entries to {out}")


if __name__ == "__main__":
    main()
