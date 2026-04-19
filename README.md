<p align="center">
  <img src="https://raw.githubusercontent.com/tuiml/tuiml/main/tuiml_logo.png" alt="TuiML Logo" width="320">
</p>
<p align="center"><strong>Machine Learning that agents can actually call.</strong></p>

<p align="center">
TuiML is an agent-native ML runtime. Install, connect to your AI agent, and start running real ML workflows &mdash; classification, regression, clustering, experiments &mdash; all from one structured interface.
</p>

<p align="center">
  <a href="https://pypi.org/project/tuiml/"><img src="https://img.shields.io/pypi/v/tuiml?style=for-the-badge" alt="PyPI version"></a>&nbsp;
  <a href="https://pypi.org/project/tuiml/"><img src="https://img.shields.io/badge/Python-≥3.10-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python versions"></a>&nbsp;
  <a href="https://tuiml.ai/docs/getting_started.html"><img src="https://img.shields.io/badge/Docs-tuiml.ai-blue?style=for-the-badge" alt="Documentation"></a>&nbsp;
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-BSD--3--Clause-blue.svg?style=for-the-badge" alt="BSD-3-Clause License"></a>&nbsp;
  <a href="https://pepy.tech/projects/tuiml"><img src="https://img.shields.io/pepy/dt/tuiml?style=for-the-badge" alt="Downloads"></a>
</p>

## Why TuiML

**Agents can call it** &mdash; Every algorithm, dataset, and metric ships with a JSON schema. Agents read the schema, call the tool, get structured results. No hallucinated parameters, no wrapper glue.

**Agents can discover it** &mdash; A queryable registry tagged by task, data shape, and benchmarks. Agents browse and pick instead of memorising class names.

**Agents can trust it** &mdash; Deterministic, typed, reproducible outputs. Every call is a loggable, replayable tool invocation you can audit, diff, and trust in production.

## Get running in 3 steps

**1. Install** &mdash; one command, installs `uv` and `tuiml` globally:

```bash
curl -fsSL https://tuiml.ai/install.sh | bash
```

Already have Python? `pip install tuiml` works too.

**2. Connect your agent** &mdash; auto-detects Claude Desktop, Cursor, Claude Code, and more:

```bash
tuiml setup
```

**3. Ask your agent** &mdash; in any connected client:

> "Train a random forest on my sales data and report the accuracy."

Your agent discovers algorithms, sets parameters from the schema, trains, evaluates, and returns structured results. No glue code.

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

## MCP Tools

The MCP server exposes 200+ tools agents can call directly. Key workflow tools include `tuiml_train`, `tuiml_predict`, `tuiml_evaluate`, `tuiml_experiment`, `tuiml_tune`, `tuiml_plot`, `tuiml_list`, `tuiml_describe`, and `tuiml_search`. Any component registered with `@classifier`, `@regressor`, or `@transformer` is automatically discoverable through these tools.

For manual setup, add this to your client's MCP config:

```json
{
    "mcpServers": {
        "tuiml": { "command": "tuiml-mcp" }
    }
}
```

## Component Registry

Browse all registered algorithms, transformers, and metrics from the local registry:

```python
from tuiml.hub import registry

classifiers = registry.list("classifier")
regressors = registry.list("regressor")
```

## Building Custom Components

Register your own algorithms and they become instantly available through the Python API, CLI, and MCP server.

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
    title={TuiML: Machine Learning that agents can actually call},
    author={Verma, Nilesh and Bifet, Albert and Pfahringer, Bernhard},
    year={2026},
    url={https://tuiml.ai}
}
```

## Links

- [Website](https://tuiml.ai)
- [Documentation](https://tuiml.ai/docs/getting_started.html)
- [API Reference](https://tuiml.ai/docs/api)
- [GitHub](https://github.com/tuiml/tuiml)
- [PyPI](https://pypi.org/project/tuiml)
- [Changelog](https://tuiml.ai/docs/changelog.html)

## Star History

<a href="https://www.star-history.com/?repos=tuiml%2Ftuiml&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/image?repos=tuiml/tuiml&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/image?repos=tuiml/tuiml&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/image?repos=tuiml/tuiml&type=date&legend=top-left" />
 </picture>
</a>
