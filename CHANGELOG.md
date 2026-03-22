# Changelog

All notable changes to TuiML will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-03-15

### Added
- 100+ machine learning algorithms across 13 categories: trees, linear, bayesian, clustering, ensemble, gradient boosting, SVM, neural, neighbors, rules, associations, time series, and anomaly detection.
- High-level API with one-liner `train()`, `predict()`, `experiment()`, and `run()` functions.
- Fluent `Workflow` builder for chainable ML pipelines with preprocessing, feature engineering, and evaluation.
- Comprehensive preprocessing module: scaling, encoding, imputation, discretization, outlier handling, SMOTE, and text vectorizers.
- Feature engineering: PCA, random projection, univariate selection, sequential selection, variance threshold, polynomial and mathematical feature generation.
- Built-in datasets: iris, diabetes, wine, breast cancer, glass, ionosphere, soybean, and segment challenge.
- Dataset generators for classification (Agrawal, Hyperplane, LED, Random RBF), clustering (blobs, circles, moons, swiss roll), and regression (Friedman, Mexican Hat, Sine).
- Data loaders for ARFF, CSV, JSON, Excel, Parquet, NumPy, and pandas formats.
- LLM integration via Model Context Protocol (MCP) server with 200+ tools for agentic ML workflows.
- CLI for training, prediction, evaluation, experiments, model serving, and hub operations.
- WekaHub community platform for algorithm discovery, publishing, and benchmarking.
- Dataset Hub for browsing, downloading, and sharing datasets with the community.
- 21 interactive Jupyter tutorials across 6 learning tracks: Quickstart, ML Simplified, LLM Friendly, Community, Deploy, and Case Studies.
- Full API documentation generated from NumPy-style docstrings with KaTeX math rendering.
- Base classes (`Classifier`, `Regressor`) with scikit-learn compatible fit/predict interface and `@classifier`/`@regressor` decorators.
- Model serialization via joblib with save/load utilities.
- Cross-validation, grid search, and hyperparameter tuning support.

[0.1.1]: https://github.com/TechyNilesh/tuiml/releases/tag/v0.1.1
