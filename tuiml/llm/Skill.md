---
name: TuiML ML
description: Machine learning toolkit - train, evaluate, and compare models using 200+ algorithms, preprocessors, and datasets
version: 0.1.2
mcp_server: tuiml-mcp
---

# TuiML Framework Guide

TuiML is a Python ML framework with 200+ components across algorithms, preprocessing, feature engineering, evaluation, and datasets. It provides three API levels: high-level one-liners, mid-level workflows, and low-level OOP.

---

## 1. Getting Started

### Install

One-liner (recommended — installs `uv` if missing, pulls the latest TuiML from GitHub, builds C++ extensions):

```bash
curl -fsSL https://tuiml.ai/install.sh | bash
```

Alternatives:

```bash
uv tool install tuiml        # isolated tool venv (recommended if uv is already installed)
pip install tuiml            # inside any existing Python environment
```

### Wire TuiML into your AI client

```bash
tuiml setup          # opens an Auto / Manual / Quit menu — default Auto
tuiml setup -y       # skip the menu and configure every detected client
tuiml setup --manual # prompt per-client (old one-by-one behaviour)
tuiml setup --list   # just show what's detected without writing anything
```

Detects Claude Desktop, Claude Code (skill file), OpenClaw, Cursor, ChatGPT Desktop, Perplexity Desktop, Codex CLI, Zed, Continue (VS Code), VS Code Copilot, Windsurf, and Goose.

### Update / uninstall

```bash
tuiml self_update         # via the MCP tool — tells the agent `restart_required: true`
uv tool install --reinstall --force tuiml   # from the shell
tuiml uninstall           # remove TuiML from every wired AI client
                          # (does NOT remove the Python package itself)
uv tool uninstall tuiml   # finally, remove the package
```

### Three API Levels

```python
# High-level: one-liner
from tuiml import train
result = train("RandomForestClassifier", "iris", target="class", cv=10)

# Mid-level: chainable workflow
from tuiml import Workflow
result = (Workflow()
    .data("iris", target="class")
    .preprocess("SimpleImputer", "StandardScaler")
    .algorithm("RandomForestClassifier", n_estimators=100)
    .evaluate(cv=10, metrics=["accuracy_score", "f1_score"])
    .run())

# Low-level: full OOP control
from tuiml.algorithms.trees import RandomForestClassifier
from tuiml.datasets import load_iris
from tuiml.evaluation import accuracy_score, train_test_split

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.X, data.y, test_size=0.2)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
print(accuracy_score(y_test, clf.predict(X_test)))
```

---

## 2. High-Level API

All top-level functions are importable from `tuiml` directly.

```python
from tuiml import (
    train, run, predict, evaluate, experiment,
    save, load, list_algorithms, describe_algorithm, search_algorithms,
    serve, stop_server, server_status,
    PRESETS,
    Workflow, WorkflowResult,
    registry, ComponentType,
)
```

### train()

```python
result = train(
    algorithm="RandomForestClassifier",
    data="iris",                          # built-in name or file path
    target="class",
    cv=10,
    algorithm_params={"n_estimators": 100, "max_depth": 10},
    preprocessing=["SimpleImputer", "StandardScaler"],
    feature_selection={"name": "SelectKBestSelector", "k": 10},
    metrics=["accuracy_score", "f1_score"],
)
# result contains model_id, metrics, model object
```

### experiment()

```python
result = experiment(
    algorithms=["RandomForestClassifier", "SVC", "NaiveBayesClassifier"],
    data="iris",
    target="class",
    cv=10,
    metrics=["accuracy_score", "f1_score", "precision_score"],
)
```

### predict() / evaluate()

```python
predictions = predict(model_id="a1b2c3", data="test_data.csv")
metrics = evaluate(model_id="a1b2c3", data="test.csv", target="class")
```

### serve()

```python
url = serve(model_id="a1b2c3", port=8000)
# Model API at http://127.0.0.1:8000/predict
stop_server(port=8000)
```

### Discovery

```python
algorithms = list_algorithms(category="algorithm")
info = describe_algorithm("RandomForestClassifier")
results = search_algorithms("ensemble")
```

---

## 3. Algorithms

13 algorithm families with exact class names as imports.

```python
from tuiml.algorithms import (
    # Bayesian
    NaiveBayesClassifier, NaiveBayesMultinomialClassifier,
    BayesianNetworkClassifier, GaussianProcessesRegressor,
    # Trees
    C45TreeClassifier, RandomForestClassifier,
    RandomTreeClassifier, DecisionStumpClassifier,
    ReducedErrorPruningTreeClassifier, HoeffdingTreeClassifier,
    M5ModelTreeRegressor, LogisticModelTreeClassifier,
    # Neighbors
    KNearestNeighborsClassifier, KStarClassifier,
    LocallyWeightedLearningRegressor,
    # Linear
    LogisticRegression, LinearRegression, SGDClassifier,
    SimpleLinearRegression, SimpleLogisticClassifier,
    # SVM
    SVC, SVR,
    # Neural
    MultilayerPerceptronClassifier, VotedPerceptronClassifier,
    # Rules
    ZeroRuleClassifier, OneRuleClassifier, RIPPERClassifier,
    PARTClassifier, DecisionTableClassifier, M5ModelRulesRegressor,
    # Ensemble
    BaggingClassifier, AdaBoostClassifier, StackingClassifier,
    VotingClassifier, RandomCommitteeClassifier,
    RandomSubspaceClassifier, LogitBoostClassifier,
    FilteredClassifier, MultiClassClassifier,
    AdditiveRegressionRegressor, RegressionByDiscretizationRegressor,
    # Gradient Boosting
    XGBoostClassifier, CatBoostClassifier, LightGBMClassifier,
    # Clustering
    KMeansClusterer, DBSCANClusterer, AgglomerativeClusterer,
    GaussianMixtureClusterer, CanopyClusterer, CobwebClusterer,
    FarthestFirstClusterer, FilteredClusterer,
    # Associations
    AprioriAssociator, FPGrowthAssociator, ECLATAssociator,
    # Anomaly
    IsolationForestDetector, LocalOutlierFactorDetector,
    EllipticEnvelopeDetector, OneClassSVMDetector, ABODDetector,
    # Time Series
    ARIMA, ExponentialSmoothing, STLDecomposition,
    AutoRegressive, MovingAverage, ARMA, Prophet,
)
```

### Usage Pattern

All algorithms follow the same fit/predict interface:

```python
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)     # classifiers only

# Fitted attributes (trailing underscore convention)
clf.classes_
clf.n_features_
clf.oob_score_       # algorithm-specific

# Static metadata
schema = RandomForestClassifier.get_parameter_schema()
caps = RandomForestClassifier.get_capabilities()
```

### Clustering (unsupervised)

```python
km = KMeansClusterer(n_clusters=3)
km.fit(X)
labels = km.predict(X)
```

### Association Rules

```python
assoc = AprioriAssociator(min_support=0.3, min_confidence=0.7)
assoc.fit(transactions)
rules = assoc.association_rules_   # list of AssociationRule objects
```

### Time Series

```python
model = ARIMA(order=(1, 1, 1))
model.fit(X_time_series, y_time_series)
forecast = model.predict(n_steps=10)
```

---

## 4. Datasets

### Built-in Datasets

```python
from tuiml.datasets import (
    load_iris, load_diabetes, load_breast_cancer,
    load_glass, load_ionosphere, load_vote, load_credit,
    load_weather, load_soybean, load_labor, load_hypothyroid,
    load_segment, load_unbalanced, load_contact_lenses,
    load_cpu, load_airline,
    load_supermarket, load_reuters_corn,
    list_datasets, load_dataset, get_dataset_info, get_datasets_by_task,
    DATASET_REGISTRY,
)

data = load_iris()
X, y = data.X, data.y
print(data.feature_names, data.n_samples, data.n_features)

# Load by name string
data = load_dataset("iris")

# Browse
all_datasets = list_datasets()
classification_sets = get_datasets_by_task("classification")
info = get_dataset_info("iris")
```

### File Loaders

```python
from tuiml.datasets import (
    load, save,                                # auto-detect by extension
    load_csv, save_csv,
    load_arff, save_arff,
    load_excel, save_excel, load_excel_sheets,
    load_parquet, save_parquet, load_parquet_partitioned,
    load_json, save_json, load_jsonl, save_jsonl, load_json_nested,
    load_numpy, save_numpy,
    load_pandas, from_pandas, to_pandas,
    Dataset,
)

data = load("data.csv")            # auto-detect
data = load_csv("data.csv")        # explicit
data = load_arff("data.arff")
data = load_parquet("data.parquet")

# Pandas interop
df = to_pandas(data)
data = from_pandas(df, target_column="class")

# Save
save(data, "output.csv")
save_parquet(data, "output.parquet")
```

### Synthetic Data Generators

```python
from tuiml.datasets import (
    RandomRBF, Agrawal, LED, Hyperplane,       # classification
    Friedman, MexicanHat, Sine,                 # regression
    Blobs, Moons, Circles, SwissRoll,           # clustering
)

data = Blobs(n_samples=1000, n_clusters=5, random_state=42).generate()
X, y = data.X, data.y
```

### Dataset Object

All loaders return a `Dataset` with:

```python
data.X              # np.ndarray (n_samples, n_features)
data.y              # np.ndarray (n_samples,) or None
data.feature_names  # list of column names
data.target_names   # list of class labels (classification)
data.name           # dataset name
data.description    # metadata string
data.n_samples      # row count
data.n_features     # column count
data.shape          # (n_samples, n_features)
```

---

## 5. Preprocessing

All preprocessors follow the fit/transform pattern. Import from `tuiml.preprocessing`.

### Scaling

```python
from tuiml.preprocessing import StandardScaler, MinMaxScaler, CenterScaler

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
```

### Imputation

```python
from tuiml.preprocessing import SimpleImputer, KNNImputer

imputer = SimpleImputer(strategy="mean")   # mean, median, most_frequent
X_clean = imputer.fit_transform(X_train)
```

### Encoding

```python
from tuiml.preprocessing import (
    OneHotEncoder, OrdinalEncoder, LabelEncoder,
    RareCategoryEncoder,
)

encoder = OneHotEncoder()
X_enc = encoder.fit_transform(X_train)
```

### Discretization

```python
from tuiml.preprocessing import (
    EqualWidthDiscretizer, QuantileDiscretizer, MDLDiscretizer,
)

disc = QuantileDiscretizer(n_bins=10)
X_disc = disc.fit_transform(X_train)
```

### Sampling & Class Balancing

```python
from tuiml.preprocessing import (
    SMOTESampler, BorderlineSMOTESampler, ADASYNSampler,
    RandomOverSampler, RandomUnderSampler,
    TomekLinksSampler, NearMissSampler,
    ClassBalanceSampler, ReservoirSampler,
)

smote = SMOTESampler(k_neighbors=5)
X_bal, y_bal = smote.fit_resample(X_train, y_train)
```

### Outliers

```python
from tuiml.preprocessing import IQROutlierDetector, ValueClipper

detector = IQROutlierDetector(multiplier=1.5)
X_clean = detector.fit_transform(X_train)
```

### Text

```python
from tuiml.preprocessing import (
    WordTokenizer, NGramTokenizer, RegexTokenizer,
    CountVectorizer, TfidfVectorizer, HashingVectorizer,
    TextCleaner, StopWordRemover, Stemmer,
)

tfidf = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf.fit_transform(documents)
```

### Time Series

```python
from tuiml.preprocessing import LagTransformer, DifferenceTransformer

lag = LagTransformer(n_lags=5)
X_lagged = lag.fit_transform(X_ts)
```


---

## 6. Feature Engineering

### Selection

```python
from tuiml.features.selection import (
    SelectKBestSelector, SelectPercentileSelector,
    SelectThresholdSelector, SelectFprSelector,
    CFSSelector, SequentialFeatureSelector,
    VarianceThresholdSelector, RandomSubsetSelector,
    BootstrapFeaturesSelector,
)

selector = SelectKBestSelector(score_func=information_gain, k=10)
X_sel = selector.fit_transform(X_train, y_train)
indices = selector.get_support(indices=True)
```

### Extraction

```python
from tuiml.features.extraction import PCAExtractor, RandomProjectionExtractor

pca = PCAExtractor(n_components=0.95)    # 95% variance retained
X_pca = pca.fit_transform(X_train)
```

### Generation

```python
from tuiml.features.generation import (
    PolynomialFeaturesGenerator, MathematicalFeaturesGenerator,
)

poly = PolynomialFeaturesGenerator(degree=2)
X_poly = poly.fit_transform(X_train)
```

---

## 7. Evaluation

Everything is importable from `tuiml.evaluation`.

### Metrics

```python
from tuiml.evaluation import (
    # Classification
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, fbeta_score,
    precision_recall_fscore_support, matthews_corrcoef, cohen_kappa_score,
    roc_auc_score, roc_curve, auc, log_loss,
    confusion_matrix, classification_report,
    # Regression
    mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score,
    # Clustering
    silhouette_score, adjusted_rand_score,
    # Information-theoretic
    entropy, mutual_information, information_gain,
    # Base
    Metric, MetricType, AverageType,
)

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average="weighted")
cm = confusion_matrix(y_true, y_pred)
```

### Splitting

```python
from tuiml.evaluation import (
    train_test_split, cross_val_score,
    KFold, StratifiedKFold, RepeatedKFold, RepeatedStratifiedKFold,
    HoldoutSplit, StratifiedHoldoutSplit,
    LeaveOneOut, LeavePOut,
    BootstrapSplit, TimeSeriesSplit,
    GroupKFold, StratifiedGroupKFold,
    ShuffleSplit, StratifiedShuffleSplit,
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
scores = cross_val_score(model, X, y, cv=10, scoring="f1_weighted")
```

### Tuning

```python
from tuiml.evaluation import (
    GridSearchCV, RandomSearchCV,
    ParameterGrid, ParameterDistribution,
)

grid = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid={"n_estimators": [50, 100, 200], "max_depth": [5, 10, None]},
    cv=5,
    scoring="f1_weighted",
)
grid.fit(X_train, y_train)
print(grid.best_params_, grid.best_score_)
```

### Experiments

```python
from tuiml.evaluation import (
    Experiment, run_experiment,
    ExperimentConfig, ExperimentResults, ExperimentType, ValidationMethod,
)

results = run_experiment(
    models={"RF": RandomForestClassifier(), "SVM": SVC(), "NB": NaiveBayesClassifier()},
    datasets={"iris": (X, y)},
    n_folds=10,
    metrics=["accuracy", "f1_weighted"],
)
print(results.to_latex())
print(results.to_markdown())
```

### Statistics

```python
from tuiml.evaluation import (
    paired_t_test, corrected_paired_t_test, one_way_anova,
    wilcoxon_signed_rank_test, friedman_test,
    nemenyi_post_hoc, bonferroni_correction,
    holm_correction, benjamini_hochberg,
    SignificanceLevel,
)

t_stat, p_value = paired_t_test(scores_a, scores_b)
f_stat, p_value = friedman_test([scores_a, scores_b, scores_c])
```

### Visualization

```python
from tuiml.evaluation import (
    plot_roc_curve, plot_pr_curve, plot_confusion_matrix,
    plot_learning_curve, plot_critical_difference,
    plot_ranking_table, plot_boxplot_comparison, plot_heatmap,
)

plot_confusion_matrix(y_true, y_pred, filename="cm.png")
plot_roc_curve(y_true, y_proba, filename="roc.png")
plot_critical_difference(results_matrix, filename="cd.png")
```

### Reporting

```python
from tuiml.evaluation import (
    ResultMatrix, format_results,
    to_latex_table, to_html_table, to_markdown_table,
)

table = to_latex_table(result_matrix)
```

---

## 8. Building Custom Components

### Custom Algorithm

```python
from tuiml.base.algorithms import Classifier, classifier
import numpy as np

@classifier(tags=["custom"], version="1.0.0")
class MyClassifier(Classifier):
    """My custom classifier.

    Parameters
    ----------
    k : int, default=5
        Number of neighbors.
    """

    def __init__(self, k=5):
        super().__init__()
        self.k = k

    def fit(self, X, y):
        """Fit the model."""
        self.classes_ = np.unique(y)
        self.X_train_ = np.asarray(X)
        self.y_train_ = np.asarray(y)
        self._is_fitted = True
        return self

    def predict(self, X):
        """Predict class labels."""
        self._check_is_fitted()
        # prediction logic
        return predictions

# Automatically registered and discoverable via tuiml_list / tuiml_train
```

### Custom Preprocessor

```python
from tuiml.base.preprocessing import Transformer, transformer
import numpy as np

@transformer(tags=["custom", "scaling"], version="1.0.0")
class MyScaler(Transformer):
    """Custom scaler."""

    def fit(self, X, y=None):
        """Learn parameters."""
        self.mean_ = np.mean(X, axis=0)
        self._is_fitted = True
        return self

    def transform(self, X):
        """Apply transformation."""
        self._check_is_fitted()
        return X - self.mean_
```

### Custom Feature Selector

```python
from tuiml.base.features import FeatureSelector, feature_selector
import numpy as np

@feature_selector(tags=["custom"], version="1.0.0")
class MySelector(FeatureSelector):
    """Custom feature selector."""

    def __init__(self, threshold=0.01):
        self.threshold = threshold

    def fit(self, X, y=None):
        """Identify features to keep."""
        self.variances_ = np.var(X, axis=0)
        self.support_ = self.variances_ > self.threshold
        self._is_fitted = True
        return self

    def transform(self, X):
        """Return selected features."""
        self._check_is_fitted()
        return np.asarray(X)[:, self.support_]
```

### Custom Metric

```python
from tuiml.base.metrics import Metric, MetricType

class MyMetric(Metric):
    """Custom evaluation metric."""

    def __init__(self):
        super().__init__("my_metric", MetricType.CLASSIFICATION)

    def compute(self, y_true, y_pred, **kwargs):
        """Compute the metric value."""
        correct = sum(a == b for a, b in zip(y_true, y_pred))
        return correct / len(y_true)
```

### Base Classes Reference

| Component | Base Class | Decorator | Required Methods |
|-----------|-----------|-----------|-----------------|
| Classifier | `Classifier` | `@classifier` | `fit()`, `predict()` |
| Regressor | `Regressor` | `@regressor` | `fit()`, `predict()` |
| Clusterer | `Clusterer` | `@clusterer` | `fit()`, `predict()` |
| Associator | `Associator` | `@associator` | `fit()` |
| Feature Selector | `FeatureSelector` | `@feature_selector` | `fit()`, `transform()` |
| Feature Extractor | `FeatureExtractor` | `@feature_extractor` | `fit()`, `transform()` |
| Feature Constructor | `FeatureConstructor` | `@feature_constructor` | `fit()`, `transform()` |
| Preprocessor | `Preprocessor` | `@preprocessor` | `fit()`, `transform()` |
| Transformer | `Transformer` | `@transformer` | `fit()`, `transform()` |
| Filter | `Filter` | `@filter_method` | `fit()`, `transform()` |
| Instance Transformer | `InstanceTransformer` | `@transformer` | `fit()`, `transform()` |
| Metric | `Metric` | (none) | `compute()` |

---

## 9. CLI

```bash
# Train
tuiml train RandomForestClassifier data.csv class --cv 10
tuiml train SVC data.csv class -p StandardScaler -p SimpleImputer
tuiml train NaiveBayesClassifier data.csv class --params '{"use_kernel_estimator": true}'

# List & search
tuiml list
tuiml list --type classifier --search "forest"
tuiml list --format json

# Predict & evaluate
tuiml predict model.pkl data.csv
tuiml evaluate RandomForestClassifier data.csv class --cv 10

# Experiment
tuiml experiment --models RF SVC NB --datasets iris.csv --n-folds 10

# Serve
tuiml serve model.pkl --port 8000

# Setup / uninstall (MCP-client wiring)
tuiml setup                     # Auto/Manual menu
tuiml setup -y                  # configure all detected clients
tuiml uninstall                 # unwire every client

# Datasets
tuiml datasets list
tuiml datasets search "classification"
tuiml datasets info iris
```

---

## 10. Local Registry

```python
from tuiml.hub import registry, ComponentType

# Local, in-process registry (no network calls)
classifiers = registry.list("classifier")
model = registry.create("RandomForestClassifier", n_estimators=100)
exists = registry.exists("RandomForestClassifier")
```

The remote community hub is currently decommissioned — use agent-authored algorithms (section 11 + 13) to add algorithms to the registry at runtime instead.

---

## 11. MCP Server (LLM Integration)

### Setup

Let the CLI wire every detected client for you:

```bash
tuiml setup -y
```

Run the server manually (for debugging) with:

```bash
tuiml-mcp
```

If you prefer editing the client config by hand, add the following to any MCP client's `mcpServers` block (Claude Desktop, Cursor, Windsurf, ChatGPT Desktop, Perplexity Desktop, Continue):

```json
{
    "mcpServers": {
        "tuiml": { "command": "tuiml-mcp" }
    }
}
```

OpenClaw uses the key `mcp.servers`, Zed uses `context_servers`, Codex CLI uses a TOML `[mcp_servers.tuiml]` block, and Goose takes YAML. `tuiml setup` handles all of these — the JSON above is shown for reference only.

### MCP Tools

**Core workflow**

| Tool | Purpose |
|------|---------|
| `tuiml_train` | Train any model with preprocessing and CV |
| `tuiml_predict` | Predict using model_id or model path |
| `tuiml_evaluate` | Evaluate model with metrics |
| `tuiml_experiment` | Compare multiple algorithms across datasets |
| `tuiml_tune` | Grid or random search over hyperparameters |
| `tuiml_statistical_test` | Friedman / Wilcoxon test on experiment results |

**Data & preparation**

| Tool | Purpose |
|------|---------|
| `tuiml_upload_data` | Upload CSV/ARFF content for other tools |
| `tuiml_read_data` | Preview rows from a dataset |
| `tuiml_data_profile` | Summary stats: shape, dtypes, missingness, cardinality |
| `tuiml_generate_data` | Generate synthetic datasets (blobs, moons, Friedman, …) |
| `tuiml_preprocess` | Apply preprocessors without a full workflow |
| `tuiml_select_features` | Feature selection as a standalone step |
| `tuiml_plot` | Standard plots (confusion matrix, ROC, PCA, …) |

**Discovery**

| Tool | Purpose |
|------|---------|
| `tuiml_list` | List components by category |
| `tuiml_describe` | Get parameter schema for any component |
| `tuiml_search` | Search components by keyword |

**Serving**

| Tool | Purpose |
|------|---------|
| `tuiml_save_model` | Save trained model to custom path |
| `tuiml_serve_model` | Start REST API for a model |
| `tuiml_stop_server` | Stop a serving server |
| `tuiml_server_status` | Check server status |

**Self-introspection & upgrade**

| Tool | Purpose |
|------|---------|
| `tuiml_system_info` | Installed version, install method, package path, latest PyPI version, update_available flag |
| `tuiml_self_update` | Upgrade to the latest release (auto-detects `uv tool` vs `pip`). Restart the client afterward. |

**Agent-authored algorithms (requires `TUIML_ALLOW_USER_ALGORITHMS=1`)**

| Tool | Purpose |
|------|---------|
| `tuiml_algorithm_skeleton` | Return a fill-in-the-blanks classifier/regressor template |
| `tuiml_create_algorithm` | AST-validate and register new Python source as a named + versioned algorithm |
| `tuiml_list_user_algorithms` | List every user algorithm and its pinned version aliases |
| `tuiml_delete_user_algorithm` | Remove a version (or all versions) from disk |

### Auto-Discovery

Any component registered with `@classifier`, `@regressor`, `@transformer`, etc. is automatically discoverable through all MCP tools. No tool definitions need updating. This is how agent-authored algorithms become first-class citizens of `tuiml_train` / `tuiml_experiment` the moment they are registered.

### Keeping this skill fresh

This `SKILL.md` is bundled with the `tuiml` package — its `version:` frontmatter matches the installed package version. Refresh workflow:

```bash
# 1. Upgrade the package (or call tuiml_self_update from the agent)
uv tool install --reinstall --force tuiml

# 2. Re-run setup to copy the bundled skill into each client's skills dir
tuiml setup -y
```

For pure MCP clients (Claude Desktop, Cursor, OpenClaw, …) the tool schemas are fetched live from the MCP server on every connection — no skill file to refresh.

### MCP Tool Examples

Train with preprocessing:

```json
{
    "tool": "tuiml_train",
    "arguments": {
        "algorithm": "SVC",
        "data": "data.csv",
        "target": "label",
        "preprocessing": [
            {"name": "SimpleImputer", "strategy": "median"},
            "StandardScaler"
        ],
        "feature_selection": {"name": "SelectKBestSelector", "k": 10},
        "cv": 10
    }
}
```

Compare algorithms:

```json
{
    "tool": "tuiml_experiment",
    "arguments": {
        "algorithms": ["RandomForestClassifier", "SVC", "NaiveBayesClassifier"],
        "data": "iris",
        "target": "class",
        "cv": 10,
        "metrics": ["accuracy_score", "f1_score"]
    }
}
```

Serve model as API:

```json
{
    "tool": "tuiml_serve_model",
    "arguments": {
        "model_id": "a1b2c3d4",
        "port": 8000
    }
}
```

### Programmatic Tool Execution

```python
from tuiml.llm import execute_tool, get_tools_for_llm

result = execute_tool("tuiml_train", algorithm="RandomForestClassifier", data="iris", target="class")
tools = get_tools_for_llm(format="mcp")
```

### MCP Resources

Datasets are available as MCP resources:
- URI: `tuiml://dataset/{name}` (e.g., `tuiml://dataset/iris`)

### Error Handling

All tools return structured responses:

```json
{
    "status": "success",
    "error": "message (if status=error)",
    "suggestion": "recovery hint",
    "recovery_tool": "tool to call",
    "recovery_params": {}
}
```

---

## 12. Common Patterns

### Full Pipeline with Cross-Validation

```python
from tuiml.algorithms.trees import RandomForestClassifier
from tuiml.preprocessing import SimpleImputer, StandardScaler
from tuiml.features.selection import CFSSelector
from tuiml.evaluation import KFold, accuracy_score
import numpy as np

kf = KFold(n_splits=5, shuffle=True)
scores = []

for train_idx, test_idx in kf.split(X):
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    # Preprocessing (fit on train only)
    imp = SimpleImputer(strategy="mean")
    scl = StandardScaler()
    sel = CFSSelector()

    X_tr = sel.fit_transform(scl.fit_transform(imp.fit_transform(X_tr)), y_tr)
    X_te = sel.transform(scl.transform(imp.transform(X_te)))

    # Train and evaluate
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_tr, y_tr)
    scores.append(accuracy_score(y_te, clf.predict(X_te)))

print(f"CV Accuracy: {np.mean(scores):.3f} +/- {np.std(scores):.3f}")
```

### Imbalanced Classification

```python
from tuiml.preprocessing import SMOTESampler

unique, counts = np.unique(y_train, return_counts=True)
print(f"Class distribution: {dict(zip(unique, counts))}")

smote = SMOTESampler(k_neighbors=5)
X_bal, y_bal = smote.fit_resample(X_train, y_train)

clf = RandomForestClassifier()
clf.fit(X_bal, y_bal)
```

### Hyperparameter Tuning

```python
from tuiml.evaluation import GridSearchCV

grid = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid={"n_estimators": [50, 100, 200], "max_depth": [5, 10, None]},
    cv=5,
    scoring="f1_weighted",
)
grid.fit(X_train, y_train)
best = grid.best_estimator_
```

### Auto-Research Loop (agent-authored algorithms)

**When to use:** the agent has an algorithmic idea ("what if I bag shallow decision trees with bootstrap weights proportional to label noise?") and wants to implement, run, and compare it against shipped algorithms — all without leaving the conversation.

**Prerequisite:** export `TUIML_ALLOW_USER_ALGORITHMS=1` in the shell that launches the MCP server, then restart the client. Source is AST-filtered, not sandboxed — the trust model is that the agent is local.

**The loop**

```
1. tuiml_algorithm_skeleton(kind="classifier")        →  template source
2. <edit fit() / predict() / __init__ hyperparams>
3. tuiml_create_algorithm(name="NoisyTreeBag",
                          kind="classifier",
                          code=<edited source>,
                          version="1.0.0")
   →  registered as NoisyTreeBag (latest)
                 and NoisyTreeBag_v1_0_0 (pinned)

4. tuiml_train(algorithm="NoisyTreeBag",
               data="iris", target="target", cv=5)
   →  baseline score

5. <iterate: tweak hyperparameters, rewrite>
6. tuiml_create_algorithm(name="NoisyTreeBag",
                          code=<new source>, version="1.0.1")
   →  NoisyTreeBag (latest now v1.0.1)
                 and NoisyTreeBag_v1_0_1 (pinned)

7. tuiml_experiment(
      algorithms=["NoisyTreeBag_v1_0_0",
                  "NoisyTreeBag_v1_0_1",
                  "RandomForestClassifier",
                  "XGBoostClassifier"],
      data=["iris", "wine", "breast_cancer"],
      target="target", cv=10,
      metrics=["accuracy_score", "f1_score"])
   →  ranked comparison with mean ± std per dataset

8. tuiml_statistical_test(
      results=<experiment output>,
      test="friedman",  post_hoc="nemenyi")
   →  tells you whether your variant is significantly better
```

**Versioning rules**

- `name` must be a valid Python identifier — usually equal to the class name.
- `version` must be semver (`MAJOR.MINOR.PATCH`). Bump it on every change.
- Every version is kept on disk at `~/.tuiml/user_algorithms/<name>/<version>/algorithm.py` with a `metadata.json` next to it (class name, kind, source hash, description). Nothing is deleted until `tuiml_delete_user_algorithm` is called.
- The bare class name (`NoisyTreeBag`) always resolves to the most recently registered version. Pinned aliases (`NoisyTreeBag_v1_0_0`) resolve to exact versions — use these when comparing variants in one `tuiml_experiment`.
- All versions are re-registered at MCP server startup, so agent work survives restarts.

**Guardrails enforced by `tuiml_create_algorithm`**

- Forbidden imports: `subprocess`, `socket`, `os`, `shutil`, `urllib`, `requests`, `httpx`, `http`, `ftplib`, `smtplib`, `ctypes`, `webbrowser`, `pty`.
- Forbidden calls: `eval`, `exec`, `compile`, `__import__`, `open`, `input`.
- Must contain at least one `@classifier` or `@regressor` decorated class.
- Declared `kind` must match the base class of the imported class.

Any rejection returns `status: error, error_type: UnsafeSource` with the specific rule that fired. Bump the version instead of overwriting when you want to keep the history.

### Model Comparison with Statistical Testing

```python
from tuiml.evaluation import run_experiment, friedman_test, nemenyi_post_hoc

results = run_experiment(
    models={"RF": RandomForestClassifier(), "SVM": SVC(), "NB": NaiveBayesClassifier()},
    datasets={"iris": (X, y)},
    n_folds=10,
)

# Friedman test for overall significance
f_stat, p_val = friedman_test(results.score_matrix)

# Nemenyi post-hoc for pairwise differences
nemenyi_post_hoc(results.score_matrix, alpha=0.05)

# Publication-ready output
print(results.to_latex())
print(results.to_markdown())
```

### Preprocessing Order

Recommended pipeline order: Imputation -> Scaling -> Encoding -> Sampling -> Feature Selection

### Naming

- Always use exact class names: `RandomForestClassifier`, not `random_forest`
- Case-sensitive: `NaiveBayesClassifier`, not `naivebayes`

### Metric Compatibility

- Classification: `accuracy_score`, `f1_score`, `precision_score`, `recall_score`, `roc_auc_score`
- Regression: `r2_score`, `mean_squared_error`, `mean_absolute_error`, `root_mean_squared_error`
- Clustering: `silhouette_score`, `adjusted_rand_score`
- If `metrics` is omitted in high-level API, defaults are selected automatically

---

## When to Use This Skill

Use TuiML when user mentions:
- Machine learning, ML, AI, data science
- Train, build, create model/classifier/predictor
- Compare, benchmark, evaluate algorithms
- Preprocessing, feature selection, data cleaning
- Cross-validation, metrics, accuracy
- Specific algorithms (random forest, SVM, neural network, etc.)
- Imbalanced data, SMOTE, oversampling
- Clustering, classification, regression
- Time series, ARIMA, forecasting
- Load/save datasets, CSV, ARFF, Parquet
- Hyperparameter tuning, grid search
- Statistical testing, Friedman, Nemenyi
- Model serving, REST API
