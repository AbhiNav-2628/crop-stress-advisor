# Verification and Test Results

Recorded on **2026-10-25**. The environment was Windows 11 Home, Python 3.13.6 (64-bit), and the
dependency versions pinned in `requirements.txt`. Every result below was actually executed. Nothing
is estimated.

## 1. Automated test suite
Command: `.\run_tests.bat` (equivalent to `python -m pytest`)

```
============================= 108 passed in 8.92s =============================
```

| File | Tests | What it covers |
|---|---|---|
| `tests/test_vegetation_indices.py` | 12 | NDVI/NDRE/GNDVI/SAVI/EVI formulas against hand-computed values; zero denominators → NaN (never inf); NaN inputs; EVI out-of-range; invalid pixels masked in every index; missing band skips only dependent indices; feature order and values; few-observation warning |
| `tests/test_validation.py` | 19 | Band mapping (missing Red/NIR, out-of-range, duplicates, unprovided bands); misaligned band files; multiband file given as single band; DN detection and scale factor; nodata and NaN masking; all-invalid input; CSV missing column, aliases and grouping, non-numeric column, partly bad values, empty and header-only files, malformed CSV; non-TIFF file; pixel-interleaved TIFF; bundled GeoTIFF nodata/georeference tags |
| `tests/test_classifier.py` | 21 | Bundled artifact valid and labelled synthetic; softmax stability; probabilities sum to 1; contributions reconstruct the logit exactly; NumPy inference matches scikit-learn; both classifiers on clear examples of all 5 classes; reproducible predictions; missing, corrupt and wrong-feature-order artifacts (fallback to rules); inconclusive for low valid fraction, missing feature, out-of-distribution, low confidence, no rule match, non-crop surface; full rule trace |
| `tests/test_advisory.py` | 12 | Bundled rules valid with verification steps; category- and crop-specific matching; priority ordering; inconclusive and poor-data advisories; confidence rule never fires without a confidence; missing facts never match; evidence carries actual values; **no rule contains dosing/spraying language**; engine independent of classifier mode; invalid rule configurations rejected |
| `tests/test_reporting.py` | 6 | Required report sections; JSON round-trip without NaN; inconclusive sample with reasons and advisories; CSV one row per sample with all features; PDF generated (valid `%PDF` header); unique report IDs |
| `tests/test_end_to_end.py` | 31 | Every bundled demo file through the full pipeline in **both** modes, with expected categories (14 files/samples × 2 modes); band files equal the multiband file; general advisory always present; missing red edge still computes the other indices |
| `tests/test_app.py` | 7 | Real `app.py` run headlessly with Streamlit `AppTest`: start-up; empty states; full demo workflow (load → every page → generate report) in both modes; loading the multi-sample CSV; broken file shows a validation error; deep link (`?demo=&page=&mode=`) |

## 2. Other checks
| Check | Command / method | Result |
|---|---|---|
| Python syntax | `python -m compileall -q app.py src ui scripts tests` | OK |
| Module imports | imported all 26 modules in `src/` and `ui/` | OK |
| Clean installation | Copied the project without `.venv` to an empty folder and ran `.\setup_windows.bat` | Exit code 0 in 373 s; fresh venv created |
| Tests on the clean install | `.\run_tests.bat -q` in that folder | **108 passed** |
| Launcher | `.\run_app.bat --server.headless true --server.port 8519` on the clean install | `/_stcore/health` returned `ok` |
| Server start-up | `streamlit run app.py --server.headless true` | Health `ok`, index HTTP 200 |
| Browser rendering | Headless Chrome via the DevTools protocol; 10 pages captured | See `docs/screenshots/`. Two defects found this way were fixed: an empty workflow diagram and a wrong message for quality-blocked results |
| Report export | `python scripts/run_demo_cli.py --out <folder>` | JSON (201 kB), CSV (18 kB), PDF (14 pages) written; PDF text extracted and checked |
| Demo model reproducibility | `python scripts/train_demo_model.py` | Prints "NumPy inference matches scikit-learn predict_proba." |

## 3. End-to-end demonstration output (ML mode)
`python scripts/run_demo_cli.py`, abridged:

| Input | Result | Top probability | Notes |
|---|---|---|---|
| plot_healthy.tif | healthy | 0.92 | |
| plot_water_stress.tif | water_stress | 0.96 | |
| plot_nutrient_stress.tif | nutrient_stress | 0.97 | |
| plot_nutrient_stress_uint16_x10000.tif | nutrient_stress | 0.97 | scale factor 10000 |
| plot_pest_stress.tif | pest_stress | 0.95 | |
| plot_disease_stress.tif | disease_stress | 0.75 | |
| plot_mild_ambiguous.tif | **healthy** | 0.75 | mild water stress missed; ADV-CONF-01 flags moderate confidence |
| plot_cloud_masked.tif | inconclusive | - | only 40% valid pixels |
| plot_bare_soil.tif | inconclusive | - | mean NDVI 0.21 < 0.3 (non-crop) |
| water_stress band files | water_stress | 0.96 | same as the multiband file |
| demo_points.csv P01-P05 | healthy / water / nutrient / pest / disease | 0.92 / 0.94 / 0.97 / 0.93 / 0.62 | P05 flagged moderate confidence |
| demo_points_no_red_edge.csv (×5) | inconclusive | - | ndre_mean, ndre_ndvi_ratio missing |
| invalid/missing_nir_column.csv | rejected | - | "missing required column(s) ['nir']" |
| invalid/text_in_numeric_column.csv | rejected | - | "contains no numeric values" |
| invalid/not_a_tiff.tif | rejected | - | "Could not read … as a TIFF/GeoTIFF raster" |

The rule-based mode gives the same categories on these files (asserted in `test_end_to_end.py`).

## 4. Model evaluation (synthetic only)
See `docs/model_card.md`. On the held-out synthetic test set (n = 500):
- the raw ML model scores accuracy 0.810;
- with gating, coverage is 0.794 and accuracy on decided plots is 0.884;
- the rule baseline with gating scores accuracy 0.674.

**These are not field results. No field validation has been performed.**

## 5. Not tested / known gaps
- Reading real sensor files (MicaSense, DJI, Sentinel-2 exports). Only synthetic files and
  synthetic in-memory TIFFs were read. Exotic GeoTIFF compressions need the optional rasterio
  package, which was not installed during these tests.
- Python versions other than 3.13.6 and operating systems other than Windows 11.
- Browser interaction was exercised through `AppTest` and scripted headless Chrome, not by a
  human clicking through every widget.

To re-run everything: `.\run_tests.bat` and `python scripts\run_demo_cli.py --out reports`.
