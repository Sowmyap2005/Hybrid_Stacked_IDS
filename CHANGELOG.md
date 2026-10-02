# Changelog

## 1.1.0 — 2026-10-02

- Added `src/15_export_result_tables.py` and the values behind Table 2, Table 3, Fig 5 and Fig 6 in `results/metrics/`, plus per-flow test-set predictions.
- Added `results/metrics/README.md` mapping each results file to the manuscript table or figure it supports.
- Renamed `src/14_pca_ablation_analysis.py` to `src/14_component_ablation.py` to match what it does (Table 6).
- Pinned scikit-learn 1.4.2 (the version used to save the model files).
- Corrected repository URL and author metadata in `CITATION.cff` and `.zenodo.json`; replaced the "zero-day detection" keyword with "unseen-attack evaluation".

## 1.0.0 — 2026-08-15

- Added the reproducible primary preprocessing and stacking pipeline.
- Added multiclass ROC-AUC/PR-AUC evaluation.
- Added PCA and component ablation artifacts.
- Added autoencoder loss, threshold, and repeated unseen-class experiments.
- Added SHAP feature-importance analysis.
- Added inference latency benchmarking.
- Added curated trained model artifacts and publication figures.
- Added MIT licensing, citation metadata, data-availability documentation, and repository contribution/security guidance.
