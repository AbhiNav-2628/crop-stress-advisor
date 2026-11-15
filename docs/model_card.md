# Classifier Card

## 1. Demonstration logistic-regression model (default mode `ml_demo`)
| Field | Value |
|---|---|
| Name / version | Demonstration logistic-regression crop-stress classifier, `demo-logreg-1.0` |
| Artifact | `models/demo_logreg_v1.json` (JSON coefficients; no pickle) |
| Type | StandardScaler + multinomial logistic regression (L2, C = 1.0, lbfgs) |
| Inputs | 9 plot-level features in fixed order: ndvi_mean, ndre_mean, gndvi_mean, savi_mean, evi_mean, ndvi_std, ndvi_p10, low_vigor_fraction, ndre_ndvi_ratio |
| Outputs | Probabilities over healthy, water_stress, nutrient_stress, pest_stress, disease_stress; *inconclusive* is added by gating |
| Created | 2026-10-25 by `scripts/train_demo_model.py`, scikit-learn 1.9.1 |

### Data provenance
- **Synthetic only.** 2,000 plots (400 per class) from `src/data/synthetic.py`, seed 7. The split is
  1,500 train / 500 test.
- No field observations, farmer data or third-party datasets were used.
- The scenarios encode assumed spectral and spatial stress patterns (see `docs/methodology.md` §4).

### Intended use
- Demonstrating and teaching an end-to-end, explainable crop-stress decision-support pipeline.
- A placeholder that shows the interface a real, validated model would plug into.

### Out-of-scope / prohibited uses
- Real agronomic diagnosis or treatment decisions.
- Insurance, credit, compensation, subsidy or regulatory decisions.
- Any claim of accuracy for real crops, regions, sensors or seasons.

### Evaluation evidence
Held-out **synthetic** test set, n = 500 (100 per class). These numbers only show that the model
recovers the generator's scenarios.

| Configuration | Accuracy | Macro F1 | Coverage | Accuracy on decided |
|---|---|---|---|---|
| Raw argmax | 0.810 | 0.809 | 1.000 | 0.810 |
| With uncertainty gating | 0.702 | 0.781 | 0.794 | 0.884 |

Per-class (raw):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| healthy | 0.81 | 0.85 | 0.83 |
| water_stress | 0.87 | 0.86 | 0.86 |
| nutrient_stress | 0.86 | 0.87 | 0.87 |
| pest_stress | 0.80 | 0.68 | 0.74 |
| disease_stress | 0.72 | 0.79 | 0.75 |

Permutation importance (mean accuracy drop): savi_mean 0.334, ndvi_std 0.210, gndvi_mean 0.194,
evi_mean 0.175, ndvi_p10 0.146, ndvi_mean 0.056, ndre_ndvi_ratio 0.056, low_vigor_fraction 0.049,
ndre_mean 0.008.

**Field validation: none.** Real-world accuracy is unknown.

### Known limitations
- Assumptions in the generator are recovered, not discovered.
- Pest and disease are frequently confused (31% of synthetic pest plots predicted as disease).
- Very mild stress can be missed (`data/demo/plot_mild_ambiguous.tif` → healthy, 0.75).
- Probabilities are not calibrated for real data.
- Correlated inputs make individual coefficients unstable to interpret.
- Out-of-distribution detection is a simple per-feature z-score check.

### Risks of misuse
- Treating a demo label as a diagnosis could lead to unnecessary pesticide or fertiliser use, or
  missed problems.
- Mitigations:
  - synthetic-data warnings on every page and in every report;
  - inconclusive gating;
  - advisories limited to inspection and expert consultation;
  - a test that forbids dosing language in advisory rules.

## 2. Transparent rule-based baseline (mode `rule_based`, also the fallback)
| Field | Value |
|---|---|
| Version | `rules-1.0` |
| Logic | Ordered decision list R1-R5 (see `docs/methodology.md` §5.2), thresholds in `config/settings.yaml` |
| Provenance | Thresholds set by inspecting synthetic feature distributions; not crop-calibrated |
| Synthetic evaluation (same test set, with gating) | accuracy 0.674, macro F1 0.684, coverage 0.972, accuracy on decided 0.693 |
| Explanation | Full rule trace with actual values |
| When used | Selected in the sidebar, or automatically if the ML artifact is missing or invalid |

## 3. Option A - pretrained models
No pretrained model is used. RGB leaf-image disease classifiers are incompatible with
vegetation-index features. A suitably licensed, validated multispectral crop-stress model can be
integrated by implementing `StressClassifier` (see `docs/architecture.md`, "Extending the system").
