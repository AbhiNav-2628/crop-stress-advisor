# Bundled demonstration data - SYNTHETIC

**Every file in this folder is synthetic.** None of it is a field measurement, and none of it
comes from a farmer, sensor vendor or public dataset.

## How it was generated
`python scripts/generate_demo_data.py` (seed 2025) produces every file here, using
`src/data/synthetic.py`:

- Each pixel is a linear mix of an illustrative **leaf spectrum** and a **soil spectrum**
  (blue, green, red, red edge, NIR), weighted by a canopy-cover fraction.
- Each scenario changes the leaf spectrum (healthy, chlorotic, wilted or necrotic), the canopy
  cover and the **spatial pattern**:
  - water stress: uniform;
  - nutrient stress: uniform with a gentle gradient;
  - pest: clumped patches of canopy loss;
  - disease: clumped patches of chlorotic or necrotic leaves.
- Illumination variation and Gaussian sensor noise are added.
- The endmember spectra are plausible illustrative values. They are not taken from a spectral library.

The demo files use a fixed severity (0.8) and central nuisance values (`typical=True`), so each one
is a *readable example* of its scenario. Training and evaluation data use random severities and
random nuisance variation, which is why the honest synthetic metrics in `models/` are well below
100%.

## Files
`manifest.json` lists every file with its description and expected outcome.

| File | Purpose |
|---|---|
| `plot_<scenario>.tif` | 64×64, 5-band float32 GeoTIFF (B, G, R, RE, NIR), nodata −9999, 5 cm pixels in UTM 43N (synthetic location) |
| `plot_mild_ambiguous.tif` | Very mild water stress (severity 0.15). Currently classified *healthy*, an honest example of a missed detection |
| `plot_cloud_masked.tif` | ~60% nodata. Expected: inconclusive |
| `plot_bare_soil.tif` | Bare soil. Expected: inconclusive (non-crop) |
| `plot_nutrient_stress_uint16_x10000.tif` | Reflectance × 10000 as uint16 (nodata 0). Needs scale factor 10000 |
| `bands/water_stress_<band>.tif` | The water-stress plot as five single-band files |
| `demo_points.csv` | 80 sampled pixels from each of 5 plots, with `sample_id`, `crop`, `is_synthetic`, `synthetic_label` |
| `demo_points_no_red_edge.csv` | Same without `red_edge`. Expected: NDRE skipped, classification inconclusive |
| `invalid/*` | Broken files for demonstrating validation errors |

## Rules for using this data
- Do not present results on these files as evidence that the system works on real crops.
- Keep real data in a separate folder; never mix it into `data/demo/`.
