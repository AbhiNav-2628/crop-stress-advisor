# Project Report Outline (academic submission)

**Title:** Multispectral Crop-Stress Diagnosis and Advisory Decision System
**Event / Problem ID:** Smart India Hackathon 2025 · SIH25099

## Front matter
Title page · Certificate · Declaration · Acknowledgements · Abstract (≈250 words; adapt README §1) ·
Table of contents · List of figures (use `docs/screenshots/`) · List of tables · Abbreviations
(NDVI, NDRE, GNDVI, SAVI, EVI, NIR, RE, DN, OOD, XAI, KVK)

## Chapter 1 - Introduction
1.1 Background: crop stress and precision agriculture · 1.2 Motivation · 1.3 Problem statement ·
1.4 Objectives · 1.5 Scope and explicit non-goals (no field validation, no treatment prescription) ·
1.6 Report organisation

## Chapter 2 - Literature and Background
2.1 Spectral behaviour of vegetation (chlorophyll absorption, red edge, NIR scattering) ·
2.2 Vegetation indices (cite Rouse 1974; Barnes 2000; Gitelson 1996; Huete 1988; Huete 2002) ·
2.3 ML for crop-stress detection (summarise only literature you have actually read; do not
invent citations) · 2.4 Explainable AI and abstention · 2.5 Research gap

## Chapter 3 - Requirements and Design
3.1 Functional requirements · 3.2 Non-functional requirements (offline, Windows, zero training) ·
3.3 Architecture (`docs/architecture.md` diagram) · 3.4 Data model (`SpectralSample`, `FeatureSet`,
`Prediction`, `AnalysisResult`) · 3.5 Configuration design · 3.6 UI design (page list)

## Chapter 4 - Methodology
4.1 Input conventions and validation · 4.2 Index computation and numerical safety ·
4.3 Feature engineering · 4.4 Synthetic data generation (assumptions, nuisance variation) ·
4.5 Classifiers: logistic regression and rule-based · 4.6 Uncertainty gating ·
4.7 Explanation methods · 4.8 Advisory rule engine (`docs/methodology.md`)

## Chapter 5 - Implementation
5.1 Technology stack and justification · 5.2 Module walkthrough (`docs/learning_guide.md` §3) ·
5.3 Key algorithms (pseudo-code for safe_divide, softmax inference, gating, rule matching) ·
5.4 Report generation · 5.5 Installation and deployment on Windows

## Chapter 6 - Testing and Results
6.1 Test strategy (unit, integration, end-to-end, UI) · 6.2 Test results (`docs/test_results.md`:
108 passed) · 6.3 Synthetic evaluation (tables from `docs/model_card.md`; confusion matrix
screenshot) · 6.4 Case studies on the demo files (correct, inconclusive and missed cases) ·
6.5 Discussion: what synthetic results do and do not show

## Chapter 7 - Responsible AI
7.1 Provenance and transparency · 7.2 Limitations (`docs/limitations.md`) · 7.3 Privacy and security ·
7.4 Misuse risks and mitigations · 7.5 Syllabus mapping (`docs/syllabus_mapping.md`)

## Chapter 8 - Conclusion and Future Work
8.1 Summary of contributions (software pipeline, not validated diagnosis) · 8.2 Future work
(field data, thermal, time series, zoning, calibrated models)

## References
Only sources you have read. The index references in `docs/methodology.md` are standard citations;
verify the details against the originals before submission.

## Appendices
A. Installation guide (README §7) · B. Configuration reference (`config/settings.yaml`) ·
C. Advisory rule listing (`config/advisory_rules.yaml`) · D. Sample report (export one PDF from
the app) · E. Screenshots · F. Test log
