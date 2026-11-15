# Methodology

## 1. Input assumptions
- Values are **surface reflectance** (0-1) after division by the user-supplied scale factor.
  Digital numbers are detected (99th percentile > 1.5) and rejected with guidance.
- Radiometric calibration (e.g. reflectance panel for UAV sensors, atmospheric correction for
  satellites) and co-registration of bands are **prerequisites**. The system does not perform them.
- Pixel validity: every mapped band must be finite, not equal to nodata, and within [0, 1.2].

## 2. Vegetation indices
Computed per pixel on valid pixels only (`src/features/vegetation_indices.py`):

| Index | Formula | Sensitivity (general literature) | Reference |
|---|---|---|---|
| NDVI | (NIR−R)/(NIR+R) | green biomass, canopy cover; saturates at high biomass | Rouse et al., 1974, NASA SP-351 |
| NDRE | (NIR−RE)/(NIR+RE) | canopy chlorophyll / nitrogen status, less saturation | Barnes et al., 2000, Proc. 5th Int. Conf. Precision Agriculture |
| GNDVI | (NIR−G)/(NIR+G) | chlorophyll concentration | Gitelson, Kaufman & Merzlyak, 1996, Remote Sens. Environ. 58 |
| SAVI | (NIR−R)/(NIR+R+L)·(1+L) | reduces soil-background effects, L = 0.5 | Huete, 1988, Remote Sens. Environ. 25 |
| EVI | 2.5(NIR−R)/(NIR+6R−7.5B+1) | reduces atmospheric/soil effects, less saturation | Huete et al., 2002, Remote Sens. Environ. 83 |

Numerical safety: `safe_divide` returns NaN where |denominator| < 1e-6 or an input is non-finite.
EVI values outside [−1, 1] are set to NaN. Undefined values are counted (`undefined_values`) and
excluded from statistics.

## 3. Features (one vector per plot)
| Feature | Definition | Intended signal |
|---|---|---|
| ndvi_mean … evi_mean | mean of each index over valid pixels | overall vigour / chlorophyll |
| ndvi_std | standard deviation of NDVI | spatial patchiness |
| ndvi_p10 | 10th percentile of NDVI | how weak the weakest areas are |
| low_vigor_fraction | share of pixels with NDVI < 0.5 | extent of low-vigour areas |
| ndre_ndvi_ratio | mean NDRE / mean NDVI | chlorophyll signal relative to biomass |

Spatial features need enough observations. Below 25 valid observations a warning is attached.

## 4. Synthetic data (training, evaluation, demo)
`src/data/synthetic.py` builds 64×64 plots with a linear mixing model:
`pixel = fc · leaf + (1 − fc) · soil`, where `leaf = (1 − w)·healthy_leaf + w·stressed_leaf`.

| Scenario | Stressed leaf | Canopy-cover change | Spatial pattern |
|---|---|---|---|
| healthy | - | - | smooth natural variation |
| water_stress | wilted | −0.20 × severity | uniform |
| nutrient_stress | chlorotic | −0.06 × severity | uniform with gentle gradient |
| pest_stress | slightly wilted | −0.70 × severity in patches | clumped patches (8-35% of area) |
| disease_stress | 50% necrotic + 50% chlorotic | −0.15 × severity in patches | clumped foci (10-40% of area) |

The data also has deliberate **nuisance variation**, so the task is not trivial:
- leaf spectrum ±5% per band (variety or phenology);
- soil brightness ×0.75-1.3;
- base canopy cover 0.74-0.93;
- illumination ×0.93-1.07;
- sensor noise σ = 0.006.

Severity is drawn from ranges that include mild cases. These design choices encode *textbook
expectations* (e.g. red-edge sensitivity to chlorophyll, patchiness of biotic damage). They are
**assumptions, not findings**: a model fitted to this data can only recover the assumptions.

## 5. Classifiers
### 5.1 Demonstration logistic regression (default)
- `scripts/train_demo_model.py`: 400 plots per class (seed 7), stratified 75/25 split, pipeline
  `StandardScaler → LogisticRegression(C=1.0, lbfgs, multinomial)`.
- Exported to `models/demo_logreg_v1.json` (scaler mean/scale, coefficients, intercepts, classes,
  feature order, provenance, scikit-learn version).
- App inference: `z = (x − mean)/scale`, `logits = W·z + b`, `p = softmax(logits)`. The training
  script asserts agreement with scikit-learn (atol 1e-9), and a unit test re-checks it.

### 5.2 Rule-based baseline
Decision list, first full match wins (thresholds in `config/settings.yaml`, chosen by inspecting
the synthetic feature distributions):

| Rule | Category | Conditions |
|---|---|---|
| R1 | pest_stress | ndre_ndvi_ratio ≥ 0.311 AND (ndvi_std ≥ 0.035 OR low_vigor_fraction ≥ 0.10) |
| R2 | disease_stress | ndre_ndvi_ratio < 0.311 AND (ndvi_std ≥ 0.035 OR low_vigor_fraction ≥ 0.10) |
| R3 | healthy | ndvi_mean ≥ 0.72 AND ndre_ndvi_ratio ≥ 0.28 |
| R4 | nutrient_stress | ndre_ndvi_ratio < 0.27 |
| R5 | water_stress | ndvi_mean < 0.72 AND ndre_ndvi_ratio ≥ 0.27 |
| - | inconclusive | nothing matched |

Rationale for R1 vs R2: exposed soil raises NDRE/NDVI (soil NDRE/NDVI ≈ 0.5), whereas chlorotic
lesions lower it. This holds in the synthetic model. Whether it holds in real fields is untested.

## 6. Reasoning under uncertainty (gating)
Applied in `predict_with_uncertainty()` in this order:
1. **Quality blockers** (classifier not run):
   - valid fraction < 0.5;
   - required feature missing;
   - mean NDVI < 0.3 (non-crop).
2. **Probabilistic checks** (ML only):
   - top probability < 0.60;
   - top-two margin < 0.15;
   - any |z-score| > 4 relative to training data (out-of-distribution).
3. **Rule check:** no rule matched.

Any failure makes the final category *inconclusive*. The raw answer and every reason are kept for
transparency. The synthetic evaluation shows the trade-off: coverage 0.794, accuracy on decided
plots 0.884, versus 0.810 without gating.

## 7. Explanations
- Linear contributions: `c_j = W[k, j] · z_j` for the predicted class *k*. They satisfy
  `Σ_j c_j + b_k = logit_k` exactly (tested). Interpretation: change in class score relative to an
  average synthetic training plot. Correlated indices may receive offsetting weights.
- Rule trace: every condition with its threshold, actual value and pass/fail.
- Permutation importance (global; 10 repeats on the synthetic test set) is stored with the metrics.

## 8. Advisory reasoning
Forward chaining over facts. A rule fires when the predicted category is in its scope, the crop is
in its scope, and all of its conditions hold. Missing facts never satisfy a condition. Results are
ordered by priority, then severity. Messages use cautious language ("can be consistent with")
and always include a human verification step. Authoritative sources are cited only where they
exist (FAO-56 for irrigation principles; leaf-colour-chart guidance for cereals). Everything else is
marked *illustrative*.

## 9. Evaluation protocol
- Metrics: accuracy (inconclusive counts as wrong), macro F1, per-class P/R/F1, confusion matrix,
  coverage, accuracy on decided samples.
- Data: a held-out synthetic split (n = 500). The app can also run a live evaluation on fresh
  synthetic plots (any seed) and on user-supplied labelled CSV samples.
- **No field validation exists.** See `docs/model_card.md`.
