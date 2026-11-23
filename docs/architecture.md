# Architecture

## Data flow
```
                 ┌──────────────────────── config/settings.yaml (all thresholds) ───────────────────────┐
                 │                                                                                       │
 GeoTIFF / band files / CSV                                                                              │
        │  src/data/ingestion.py        read_raster() → RasterInput      read_csv_table() → DataFrame    │
        ▼                                                                                                │
 src/data/validation.py + preprocessing.py                                                               │
        │  band mapping check · alignment check · DN detection · scale factor · nodata/range mask        │
        ▼                                                                                                │
   SpectralSample(bands, valid_mask, warnings, metadata)      ← one per plot / CSV sample_id             │
        │                                                                                                │
        ▼  src/pipeline.py: analyze_sample()                                                             │
 src/features/vegetation_indices.py   compute_indices()  → NDVI, NDRE, GNDVI, SAVI, EVI maps (NaN-safe)  │
 src/features/feature_engineering.py  extract_features() → FeatureSet (9 values, FEATURE_ORDER)          │
        │                                                                                                │
        ▼                                                                                                │
 src/models/inference.py   predict_with_uncertainty()                                                    │
        │   1. quality blockers (valid %, missing features, non-crop) ── fail → inconclusive             │
        │   2. classifier.predict()  ← CNNStressClassifier, LogisticRegressionClassifier, RuleBasedClassifier│
        │   3. confidence / margin / out-of-distribution checks ─────── fail → inconclusive              │
        ▼                                                                                                │
   Prediction(category, raw_category, confidence, probabilities, reasons, explanation, classifier info)  │
        │                                                                                                │
        ├──► src/explainability/explanations.py  explain_prediction() → saliency, contributions, Grad-CAM │
        │                                                                                                │
        └──► src/advisory/rule_engine.py  build_facts() → evaluate_rules(config/advisory_rules.yaml)     │
                                                                                                         │
   AnalysisResult ──► ui/* pages (Streamlit)   ──► src/reporting/report_generator.py → JSON / CSV / PDF ─┘
```

## Module responsibilities
| Module | Responsibility | Depends on |
|---|---|---|
| `src/config.py` | Load YAML, resolve project paths | PyYAML |
| `src/data/ingestion.py` | Read TIFFs (rasterio if installed, else tifffile) and CSVs | validation |
| `src/data/validation.py` | All input checks; `ValidationError` | NumPy, pandas |
| `src/data/preprocessing.py` | Build `SpectralSample`s, scaling, masks, CSV grouping | ingestion, validation |
| `src/data/synthetic.py` | Seeded synthetic plot generator | NumPy |
| `src/features/vegetation_indices.py` | Index formulas, `safe_divide` | NumPy |
| `src/features/feature_engineering.py` | `FEATURE_ORDER`, `extract_features`, index statistics | NumPy, pandas |
| `src/models/classifier.py` | `StressClassifier` interface, logistic and rule classifiers | features |
| `src/models/cnn_model.py` | PyTorch 1D & 2D Convolutional Neural Network architectures + Grad-CAM | PyTorch |
| `src/models/cnn_classifier.py` | `CNNStressClassifier`, saliency attributions, raster diagnosis | PyTorch, NumPy |
| `src/models/artifacts.py` | JSON model load/validate/save | features |
| `src/models/inference.py` | Choose classifier (CNN, ML, rules with fallback), uncertainty gating | classifier, artifacts |
| `src/models/evaluation.py` | Metrics (accuracy, P/R/F1, confusion, coverage) | scikit-learn |
| `src/explainability/explanations.py` | Human-readable explanation dictionary | inference |
| `src/advisory/rule_engine.py` | Facts, rule validation, matching, ordering | - |
| `src/reporting/report_generator.py` | Report dict, JSON, CSV, PDF | fpdf2, pandas |
| `src/pipeline.py` | `analyze_sample()` glue | all of the above |
| `ui/*.py` | One Streamlit page each, plus `common.py` (state, caching, badges) | src |
| `app.py` | Page config, sidebar, navigation, deep links | ui |

## Design decisions
- **One pipeline function** (`analyze_sample`) is used by the app, the CLI demo, the training
  script (through `compute_sample_features`) and the tests. Training and inference preprocessing
  therefore cannot drift apart.
- **Classifier does not decide "inconclusive"** (except a rule list with no match). The gating lives
  in `inference.py`, so every classifier gets the same safety checks.
- **Advisories are decoupled** from classifiers through a flat `facts` dictionary. A new classifier
  needs no advisory changes, and advisory rules can be edited by an agronomist in YAML.
- **JSON model artifact** instead of pickle: readable, safe to load, independent of the
  scikit-learn version. The feature order is stored in the artifact and checked at load time.
- **Graceful degradation:** missing or corrupt model → rule-based fallback with a visible notice;
  missing optional bands → affected indices skipped with reasons; rasterio absent → tifffile.
- **Configuration over code:** thresholds, presets, aliases, crops and categories are in
  `config/settings.yaml`.
- **State in the UI:** loaded samples and results live in `st.session_state`. Results are recomputed
  only when the samples, mode or crop change (`ui/common.current_results`).

## Extending the system
- *New model*: subclass `StressClassifier`, implement `predict(features) -> ClassifierOutput`,
  register it in `build_classifier()` and `MODE_LABELS`.
- *New feature*: add it to `FEATURE_ORDER` and `extract_features`, then refit the model (the
  artifact check will refuse the old model, by design).
- *New advisory*: add an entry to `config/advisory_rules.yaml`. The engine validates it on load.
- *New input format*: add a reader in `ingestion.py` that ends in a `SpectralSample`.
