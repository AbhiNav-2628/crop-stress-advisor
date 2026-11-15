# Syllabus Mapping

Status key:
- **Implemented in code**: working code you can run and point to.
- **Demonstrated conceptually**: discussed or illustrated in the app or docs, but not a separate technique in code.
- **Optional extension**: a clear hook exists, but nothing is implemented.
- **Outside current scope**: not covered.

| # | Syllabus concept | Status | Where / how |
|---|---|---|---|
| 1 | AI problem formulation | Implemented in code | Inputs, outputs, categories and constraints formalised: `config/settings.yaml` (categories), `src/features/feature_engineering.py` (feature space), `src/pipeline.py` (pipeline). README §2-4. |
| 2 | Intelligent systems / decision support | Implemented in code | Perception (indices), reasoning (classifier, gating, rules), action recommendation (advisories), human in the loop. `src/pipeline.py`. |
| 3 | Intelligent agents | Demonstrated conceptually | The pipeline follows a sense → reason → recommend cycle, but there is no autonomous agent acting in an environment. See the learning guide. |
| 4 | Supervised classification | Implemented in code | Multinomial logistic regression trained on labelled (synthetic) data: `scripts/train_demo_model.py`, `src/models/classifier.py`. |
| 5 | Feature engineering / data preprocessing | Implemented in code | Masks, scaling, indices, plot statistics: `src/data/preprocessing.py`, `src/features/*`. |
| 6 | Machine learning workflow (train/test split, scaling, fitting) | Implemented in code | Stratified split, StandardScaler, fit, export: `scripts/train_demo_model.py`. |
| 7 | Deep learning | Optional extension | Not implemented and not justified without real labelled data. A CNN could implement `StressClassifier` (`docs/architecture.md`). |
| 8 | Knowledge representation | Implemented in code | Advisory knowledge as declarative YAML rules (`config/advisory_rules.yaml`); classifier rules as `Condition`/`DecisionRule` objects. |
| 9 | Rule-based reasoning (forward chaining, production rules) | Implemented in code | `src/advisory/rule_engine.py` (facts → matching rules → prioritised conclusions); decision list in `RuleBasedClassifier`. |
| 10 | Reasoning under uncertainty | Implemented in code | Softmax probabilities, confidence and margin thresholds, out-of-distribution z-score check, explicit *inconclusive* outcome: `src/models/inference.py`. |
| 11 | Probabilistic models (Bayes nets, etc.) | Outside current scope | Logistic regression is a probabilistic discriminative model, but there are no Bayesian networks or explicit Bayesian inference. |
| 12 | Explainable AI | Implemented in code | Exact linear contributions, rule traces, permutation importance: `src/explainability/explanations.py`, `scripts/train_demo_model.py`, Evaluation page. |
| 13 | Model evaluation | Implemented in code | Accuracy, macro F1, per-class P/R/F1, confusion matrix, coverage: `src/models/evaluation.py`; live re-evaluation in the app. |
| 14 | Reproducibility | Implemented in code | Fixed seeds (demo 2025, training 7), JSON artifact with versions, pinned `requirements.txt`, deterministic inference tested in `tests/test_classifier.py`. |
| 15 | Transparency | Implemented in code | Model mode, version and provenance on every page and report; model card. |
| 16 | Fairness and bias | Demonstrated conceptually | Discussed as dataset bias (synthetic data, no crop/region coverage) in the model card and limitations. No fairness metrics, because there are no groups of people or regions in the data. |
| 17 | Privacy and security | Implemented in code (partly) | Local-only processing, telemetry disabled (`.streamlit/config.toml`), no pickle loading, input validation. Location sensitivity is documented. |
| 18 | Responsible AI / human oversight | Implemented in code | Inconclusive outputs, non-prescriptive advisories (test-enforced), mandatory verification steps, disclaimers in reports. |
| 19 | Python-based AI development | Implemented in code | NumPy, pandas, scikit-learn, Streamlit, Plotly, pytest; modular package with 108 tests. |
| 20 | Foundation models / modern AI paradigms | Outside current scope | No large language or vision foundation model is used. Possible extensions are discussed in the learning guide (e.g. LLM-generated advisory wording reviewed by experts; geospatial foundation models), but none is implemented. |
| 21 | Search / optimisation algorithms | Demonstrated conceptually | The lbfgs optimiser inside scikit-learn fits the model; there is no hand-written search algorithm. |
| 22 | Unsupervised learning (clustering) | Optional extension | Within-field zoning by clustering is listed under future scope; not implemented. |

## Short explanations for viva questions
- *Why logistic regression?* It is a supervised classifier with probabilistic outputs and exactly
  decomposable explanations. It is small enough to store as readable JSON, which suits a demo with
  synthetic data.
- *Where is the "AI"?* Learned classification (ML), symbolic rule reasoning (knowledge-based AI),
  uncertainty handling (abstention) and explanation, combined into a decision-support system.
- *Is it validated?* No. Only synthetic evaluation exists, and the app says so.
