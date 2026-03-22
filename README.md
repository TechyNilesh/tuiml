<p align="center">
  <img src="https://raw.githubusercontent.com/TechyNilesh/tuiml/main/tuiml_logo.png" alt="TuiML Logo" width="320">
</p>
<p align="center"><strong>An Open-Source Agentic ML Ecosystem</strong></p>

<p align="center">
TuiML is a complete ML ecosystem for building algorithms, training models, running experiments, and serving predictions through a clean Python API. Every algorithm is also exposed as a tool via MCP so AI agents can use them directly. A built-in community Hub lets anyone publish and discover algorithms, datasets, and models.
</p>

<p align="center">
  <a href="https://pypi.org/project/tuiml/"><img src="https://img.shields.io/pypi/v/tuiml?style=for-the-badge" alt="PyPI version"></a>&nbsp;
  <a href="https://pypi.org/project/tuiml/"><img src="https://img.shields.io/pypi/pyversions/tuiml?style=for-the-badge" alt="Python versions"></a>&nbsp;
  <a href="https://tuiml.ai/docs/getting_started.html"><img src="https://img.shields.io/badge/Docs-tuiml.ai-blue?style=for-the-badge" alt="Documentation"></a>&nbsp;
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-BSD--3--Clause-blue.svg?style=for-the-badge" alt="BSD-3-Clause License"></a>&nbsp;
  <a href="https://pepy.tech/projects/tuiml"><img src="https://static.pepy.tech/badge/tuiml?style=for-the-badge" alt="Downloads"></a>
</p>

## Three Pillars

**Simplified ML** -- High-level Python APIs designed to be beginner-friendly while remaining powerful enough for researchers to experiment and innovate.

**LLM-Friendly** -- Every algorithm is exposed as a tool via the Model Context Protocol (MCP), making them instantly accessible to any AI agent or assistant.

**Community Driven** -- Connected with TuiML Hub where anyone can push results and download state-of-the-art algorithms for benchmarking.

## Installation

```bash
pip install tuiml
```

Or with uv:

```bash
uv add tuiml
```

## Quick Start

TuiML provides three API levels. Pick the one that fits your workflow.

**High-Level** -- One-liner for quick experiments:

```python
from tuiml import train

result = train("RandomForestClassifier", "iris", target="class", cv=10)
print(f"Accuracy: {result.metrics['accuracy_score']:.3f}")
```

**Mid-Level** -- Chainable workflow:

```python
from tuiml import Workflow

result = (Workflow()
    .data("iris", target="class")
    .preprocess("SimpleImputer", "StandardScaler")
    .algorithm("RandomForestClassifier", n_estimators=100)
    .evaluate(cv=10, metrics=["accuracy_score", "f1_score"])
    .run())
```

**Low-Level** -- Full OOP control:

```python
from tuiml.algorithms.trees import RandomForestClassifier
from tuiml.datasets import load_iris
from tuiml.evaluation import accuracy_score, train_test_split

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.X, data.y, test_size=0.2)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
print(accuracy_score(y_test, clf.predict(X_test)))
```

## What's Included

TuiML ships with 13 algorithm families, many originally from Weka, completely rewritten in Python with C++ acceleration for hot paths.

| Category | Examples |
|----------|----------|
| **Trees** | RandomForestClassifier, C45TreeClassifier, HoeffdingTreeClassifier, M5ModelTreeRegressor |
| **Bayesian** | NaiveBayesClassifier, BayesianNetworkClassifier, GaussianProcessesRegressor |
| **Neighbors** | KNearestNeighborsClassifier, KStarClassifier |
| **Linear** | LogisticRegression, LinearRegression, SGDClassifier |
| **SVM** | SVC, SVR |
| **Neural** | MultilayerPerceptronClassifier, VotedPerceptronClassifier |
| **Rules** | ZeroRuleClassifier, OneRuleClassifier, RIPPERClassifier, PARTClassifier |
| **Ensemble** | BaggingClassifier, AdaBoostClassifier, StackingClassifier, VotingClassifier |
| **Gradient Boosting** | XGBoostClassifier, CatBoostClassifier, LightGBMClassifier |
| **Clustering** | KMeansClusterer, DBSCANClusterer, AgglomerativeClusterer |
| **Associations** | AprioriAssociator, FPGrowthAssociator |
| **Anomaly Detection** | IsolationForestDetector, LocalOutlierFactorDetector |
| **Time Series** | ARIMA, ExponentialSmoothing, Prophet |

Plus preprocessing (scaling, encoding, imputation, SMOTE, text vectorization), feature engineering (selection, extraction, generation), evaluation (metrics, cross-validation, tuning, statistical tests), and 15+ built-in datasets.

## MCP Server

TuiML includes an MCP server that exposes all algorithms as tools for AI agents.

```bash
tuiml-mcp
```

Configure with Claude Desktop:

```json
{
    "mcpServers": {
        "tuiml": {
            "command": "tuiml-mcp"
        }
    }
}
```

Tools available: `tuiml_train`, `tuiml_predict`, `tuiml_evaluate`, `tuiml_experiment`, `tuiml_list`, `tuiml_describe`, `tuiml_search`, `tuiml_serve_model`. Any component registered with `@classifier`, `@regressor`, or `@transformer` is automatically discoverable.

## CLI

```bash
tuiml train RandomForestClassifier data.csv class --cv 10
tuiml list --type classifier --search "forest"
tuiml experiment -a RandomForestClassifier -a SVC -d iris.csv -t class --cv 10
tuiml serve model.pkl --port 8000
```

## TuiML Hub

A community platform for sharing algorithms, datasets, and models. Visit [tuiml.ai](https://tuiml.ai) to browse.

```python
from tuiml.hub import registry, remote

classifiers = registry.list("classifier")
remote.search("random forest")
remote.install("community-algorithm-name")
```

## Building Custom Components

Register your own algorithms and they become instantly available through the Python API, CLI, MCP server, and Hub.

```python
from tuiml.base.algorithms import Classifier, classifier

@classifier(tags=["custom"], version="1.0.0")
class MyClassifier(Classifier):
    def __init__(self, k=5):
        super().__init__()
        self.k = k

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self._is_fitted = True
        return self

    def predict(self, X):
        self._check_is_fitted()
        return predictions
```

## Documentation

Full documentation is available at [tuiml.ai/docs](https://tuiml.ai/docs/getting_started.html), including getting started guides, API reference, and tutorials.

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{tuiml2026,
    title={TuiML: An Open-Source Agentic ML Ecosystem},
    author={Verma, Nilesh and Bifet, Albert and Pfahringer, Bernhard},
    year={2026},
    url={https://tuiml.ai}
}
```

## Links

- [Website](https://tuiml.ai)
- [Documentation](https://tuiml.ai/docs/getting_started.html)
- [API Reference](https://tuiml.ai/docs/api)
- [GitHub](https://github.com/TechyNilesh/TuiML)
- [PyPI](https://pypi.org/project/tuiml)
- [Changelog](https://tuiml.ai/docs/changelog.html)
