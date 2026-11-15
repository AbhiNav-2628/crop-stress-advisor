# Learning Guide and Code Walkthrough

The application already works; this guide is for understanding it afterwards.

## 1. How the modules interact (one sentence each)
1. `app.py` draws the sidebar and calls the chosen page's `render()` in `ui/`.
2. A page loads data through `src/data/ingestion.py`; `src/data/preprocessing.py` validates it
   (`validation.py`) and produces `SpectralSample` objects.
3. `ui/common.current_results()` calls `src/pipeline.analyze_sample()` for each sample.
4. `analyze_sample` computes indices, then features, then a gated prediction, then an
   explanation, then advisories, and returns an `AnalysisResult`.
5. Pages display parts of that result; `src/reporting/report_generator.py` turns results into
   JSON/CSV/PDF.

## 2. Files to study first (in this order)
| Order | File | What to learn |
|---|---|---|
| 1 | `src/features/vegetation_indices.py` | NumPy arrays, vectorised maths, NaN-safe division |
| 2 | `src/features/feature_engineering.py` | Summarising arrays into features; dataclasses |
| 3 | `src/data/validation.py` | Defensive programming, custom exceptions |
| 4 | `src/data/preprocessing.py` | Masks, scaling, grouping a pandas DataFrame |
| 5 | `src/models/classifier.py` | Classes and interfaces; softmax; logistic regression by hand; a decision list |
| 6 | `src/models/inference.py` | Uncertainty handling and abstention |
| 7 | `src/explainability/explanations.py` | Why coef × z is an exact explanation for linear models |
| 8 | `src/advisory/rule_engine.py` + `config/advisory_rules.yaml` | Production rules, forward chaining |
| 9 | `src/pipeline.py` | How everything connects |
| 10 | `scripts/train_demo_model.py` | The ML workflow: data → split → scale → fit → evaluate → export |
| 11 | `src/data/synthetic.py` | How the synthetic data encodes assumptions |
| 12 | `ui/*.py`, `app.py` | Streamlit, session state, Plotly |
| 13 | `tests/` | How each behaviour is checked |

## 3. Walkthrough of one prediction (nutrient demo)
1. `read_raster("plot_nutrient_stress.tif")` returns a (5, 64, 64) array and nodata −9999.
2. `sample_from_raster` picks bands with the mapping {blue:1, …, nir:5}, divides by the scale (1.0),
   checks that the data is not DN, and builds `valid_mask` (all True here).
3. `compute_indices` returns five 64×64 index maps.
4. `extract_features` gives, e.g., ndvi_mean ≈ 0.676, ndre_ndvi_ratio ≈ 0.245 (low, which suggests
   a chlorophyll signal).
5. `LogisticRegressionClassifier.predict` standardises the features (z), computes 5 logits and a
   softmax, giving nutrient_stress ≈ 0.97.
6. `predict_with_uncertainty`: valid 100%, no missing features, NDVI > 0.3, probability > 0.6,
   margin > 0.15, all |z| < 4, so the category is kept.
7. `explain_prediction` sorts the coef×z terms. The bars on the Explainability page add up to the
   class score.
8. `build_facts` + `evaluate_rules` match ADV-NUTR-01 (check nutrient records) and ADV-GEN-01
   (human verification).

Try it in a terminal:
```powershell
.\.venv\Scripts\python.exe scripts\run_demo_cli.py
```

## 4. Suggested learning path
1. **Python fundamentals:** functions, classes, dataclasses, exceptions, modules. Read
   `validation.py`, then `rule_engine.py`.
2. **NumPy and pandas:** broadcasting, boolean masks, NaN handling, `groupby`. Read
   `vegetation_indices.py`, then `preprocessing.py`.
3. **Remote-sensing basics:** reflectance, spectral bands, why red edge matters. Read
   `docs/methodology.md` §1-3.
4. **Machine learning:** train/test split, scaling, logistic regression, softmax, overfitting. Run
   `scripts/train_demo_model.py`, then change `C` or `PLOTS_PER_CLASS` and compare the metrics.
5. **Evaluation:** confusion matrix, precision/recall/F1, coverage vs accuracy. Use the Evaluation
   page with different seeds.
6. **Knowledge-based AI:** decision lists vs production rules. Add a rule to `advisory_rules.yaml`
   and write a test for it.
7. **Explainable AI:** prove to yourself that Σ coef·z + b = logit (see
   `test_linear_contributions_reconstruct_the_score`); compare with permutation importance; read
   about SHAP and when it is needed (non-linear models).
8. **Deep learning (next step):** learn PyTorch basics, then think about what labelled data a CNN
   on index patches would need. Implement it as a new `StressClassifier` only once such data exists.
9. **Responsible AI:** read `docs/model_card.md` and `docs/limitations.md`. Practise explaining
   why the system abstains.

## 5. Exercises
- Lower `uncertainty.min_confidence` to 0.4 in `config/settings.yaml`. What happens to coverage and
  accuracy on the Evaluation page (fresh evaluation)?
- Add a sixth feature (e.g. `ndre_std`), refit the model, and observe that the old artifact is
  rejected (feature-order check).
- Write a CSV with your own reflectance values and a `label` column, then evaluate it on the
  Evaluation page.

## 6. Foundation models: where they could (and could not) fit
Not used in this project. Possible future roles:
- an LLM that rephrases advisories into regional languages, with expert review;
- a geospatial foundation model for feature extraction, if it is licensed and validated for crop stress.

Neither would remove the need for field validation.
