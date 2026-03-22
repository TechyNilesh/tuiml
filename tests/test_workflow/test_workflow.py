"""Tests for tuiml.workflow (Workflow and WorkflowResult)."""

import numpy as np
import pytest

from tuiml.workflow import Workflow, WorkflowResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class _FakeModel:
    """Minimal model-like object for unit tests that don't need real ML."""

    classes_ = np.array([0, 1])

    def predict(self, X):
        return np.zeros(len(X))

    def predict_proba(self, X):
        return np.ones((len(X), 2)) * 0.5


class _FakeModelNoProba:
    """A fake model without predict_proba."""

    def predict(self, X):
        return np.zeros(len(X))


class _TrackingPreprocessor:
    """Training-time preprocessor used to verify split-scoped fitting."""

    fit_sizes = []

    def fit(self, X, y=None):
        self.__class__.fit_sizes.append(len(X))
        self.offset_ = np.mean(X, axis=0)
        return self

    def transform(self, X):
        return np.asarray(X) - self.offset_

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class _TrackingModel:
    """Minimal classifier that records inputs it receives."""

    fit_sizes = []

    def fit(self, X, y):
        self.__class__.fit_sizes.append(len(X))
        self.last_predict_input_ = None
        return self

    def predict(self, X):
        self.last_predict_input_ = np.asarray(X)
        return np.zeros(len(X), dtype=int)


def _make_classification_data(n_samples=100, n_features=4, seed=42):
    """Create a simple binary classification dataset."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


# ===========================================================================
# WorkflowResult tests
# ===========================================================================
class TestWorkflowResult:
    def test_init_defaults(self):
        result = WorkflowResult()
        assert result.model is None
        assert result.model_id is None
        assert result.metrics == {}
        assert result.cv_results is None
        assert result.predictions is None
        assert result.probabilities is None
        assert result.feature_importance is None
        assert result.preprocessing_pipeline is None
        assert result.metadata == {}

    def test_init_with_values(self):
        model = _FakeModel()
        metrics = {"accuracy": 0.95}
        result = WorkflowResult(
            model=model,
            model_id="test_id",
            metrics=metrics,
            metadata={"algorithm": "FakeModel"},
        )
        assert result.model is model
        assert result.model_id == "test_id"
        assert result.metrics["accuracy"] == 0.95
        assert result.metadata["algorithm"] == "FakeModel"

    def test_repr_contains_metrics(self):
        result = WorkflowResult(metrics={"accuracy": 0.9})
        repr_str = repr(result)
        assert "metrics" in repr_str
        assert "accuracy" in repr_str

    def test_predict_with_model(self):
        result = WorkflowResult(model=_FakeModel())
        X = np.array([[1, 2], [3, 4]])
        preds = result.predict(X)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == 2

    def test_predict_no_model(self):
        result = WorkflowResult()
        with pytest.raises(RuntimeError, match="No trained model available"):
            result.predict(np.array([[1, 2]]))

    def test_predict_proba_with_model(self):
        result = WorkflowResult(model=_FakeModel())
        X = np.array([[1, 2], [3, 4]])
        probas = result.predict_proba(X)
        assert probas.shape == (2, 2)

    def test_predict_proba_no_model(self):
        result = WorkflowResult()
        with pytest.raises(RuntimeError, match="No trained model available"):
            result.predict_proba(np.array([[1, 2]]))

    def test_predict_proba_not_supported(self):
        result = WorkflowResult(model=_FakeModelNoProba())
        with pytest.raises(AttributeError, match="does not support predict_proba"):
            result.predict_proba(np.array([[1, 2]]))


# ===========================================================================
# Workflow — fluent API (configuration) tests
# ===========================================================================
class TestWorkflowFluentAPI:
    def test_init_empty(self):
        wf = Workflow()
        assert wf._data is None
        assert wf._target is None
        assert wf._preprocessing_steps == []
        assert wf._feature_selection is None
        assert wf._split_config is None
        assert wf._model is None
        assert wf._model_params == {}
        assert wf._evaluation_config == {}

    def test_init_with_array(self):
        X = np.array([[1, 2], [3, 4]])
        wf = Workflow(X, target="label")
        assert wf._data is X
        assert wf._target == "label"

    def test_chaining_returns_self(self):
        wf = Workflow()
        assert wf.impute() is wf
        assert wf.normalize() is wf
        assert wf.standardize() is wf
        assert wf.encode_categorical() is wf
        assert wf.train("SomeAlgo") is wf
        assert wf.evaluate() is wf

    def test_impute_default(self):
        wf = Workflow().impute()
        assert len(wf._preprocessing_steps) == 1
        step = wf._preprocessing_steps[0]
        assert step["name"] == "SimpleImputer"
        assert step["strategy"] == "mean"

    def test_impute_knn(self):
        wf = Workflow().impute(strategy="knn")
        step = wf._preprocessing_steps[0]
        assert step["name"] == "KNNImputer"

    def test_impute_median(self):
        wf = Workflow().impute(strategy="median")
        step = wf._preprocessing_steps[0]
        assert step["strategy"] == "median"

    def test_normalize_default(self):
        wf = Workflow().normalize()
        step = wf._preprocessing_steps[0]
        assert step["name"] == "MinMaxScaler"

    def test_normalize_zscore(self):
        wf = Workflow().normalize(method="zscore")
        step = wf._preprocessing_steps[0]
        assert step["name"] == "StandardScaler"

    def test_standardize(self):
        wf = Workflow().standardize()
        step = wf._preprocessing_steps[0]
        assert step["name"] == "StandardScaler"

    def test_encode_categorical_onehot(self):
        wf = Workflow().encode_categorical(method="onehot")
        step = wf._preprocessing_steps[0]
        assert step["name"] == "OneHotEncoder"

    def test_encode_categorical_ordinal(self):
        wf = Workflow().encode_categorical(method="ordinal")
        step = wf._preprocessing_steps[0]
        assert step["name"] == "OrdinalEncoder"

    def test_select_features(self):
        wf = Workflow().select_features(k=20)
        assert wf._feature_selection is not None
        assert wf._feature_selection["name"] == "SelectKBestSelector"
        assert wf._feature_selection["k"] == 20

    def test_pca(self):
        wf = Workflow().pca(n_components=3)
        assert wf._feature_selection is not None
        assert wf._feature_selection["name"] == "PCAExtractor"
        assert wf._feature_selection["n_components"] == 3

    def test_pca_variance_ratio(self):
        wf = Workflow().pca(n_components=0.95)
        assert wf._feature_selection["n_components"] == 0.95

    def test_split(self):
        wf = Workflow().split(test_size=0.3, stratify=False, random_state=99)
        assert wf._split_config is not None
        assert wf._split_config["test_size"] == 0.3
        assert wf._split_config["stratify"] is False
        assert wf._split_config["random_state"] == 99

    def test_train(self):
        wf = Workflow().train("RandomForestClassifier", n_trees=100)
        assert wf._model == "RandomForestClassifier"
        assert wf._model_params == {"n_trees": 100}

    def test_model_alias(self):
        wf = Workflow().model("SVM", C=1.0)
        assert wf._model == "SVM"
        assert wf._model_params == {"C": 1.0}

    def test_evaluate(self):
        wf = Workflow().evaluate()
        assert wf._evaluation_config["method"] == "holdout"
        # metrics=None means auto-resolve based on algorithm type at run time
        assert wf._evaluation_config["metrics"] is None

    def test_evaluate_custom_metrics(self):
        wf = Workflow().evaluate(metrics=["precision_score"], test_size=0.3)
        assert wf._evaluation_config["metrics"] == ["precision_score"]
        assert wf._evaluation_config["test_size"] == 0.3

    def test_cross_validate(self):
        wf = Workflow().cross_validate(cv=5)
        assert wf._evaluation_config["method"] == "cross_validate"
        assert wf._evaluation_config["cv"] == 5

    def test_cross_validate_default(self):
        wf = Workflow().cross_validate()
        assert wf._evaluation_config["cv"] == 10

    def test_handle_missing_alias(self):
        """handle_missing() should behave identically to impute()."""
        wf = Workflow().handle_missing(strategy="median")
        step = wf._preprocessing_steps[0]
        assert step["name"] == "SimpleImputer"
        assert step["strategy"] == "median"

    def test_preprocess_generic(self):
        wf = Workflow().preprocess("MyCustomScaler", alpha=0.5)
        step = wf._preprocessing_steps[0]
        assert step["name"] == "MyCustomScaler"
        assert step["alpha"] == 0.5

    def test_resample(self):
        wf = Workflow().resample(method="smote")
        step = wf._preprocessing_steps[0]
        assert step["name"] == "SMOTESampler"

    def test_load_method(self):
        wf = Workflow()
        X = np.array([[1, 2]])
        assert wf.load(X) is wf
        assert wf._data is X


# ===========================================================================
# Workflow — to_config
# ===========================================================================
class TestWorkflowConfig:
    def test_to_config(self):
        config = (
            Workflow("data.csv", target="class")
            .impute(strategy="mean")
            .standardize()
            .select_features(k=5)
            .split(test_size=0.25)
            .train("NaiveBayesClassifier", alpha=1.0)
            .evaluate()
            .to_config()
        )

        assert config["data"]["source"] == "data.csv"
        assert config["data"]["target"] == "class"
        assert len(config["preprocessing"]) == 2
        assert config["feature_selection"]["name"] == "SelectKBestSelector"
        assert config["split"]["test_size"] == 0.25
        assert config["model"]["name"] == "NaiveBayesClassifier"
        assert config["model"]["params"]["alpha"] == 1.0
        assert config["evaluation"]["method"] == "holdout"

    def test_to_config_minimal(self):
        config = Workflow().train("X").to_config()
        assert config["model"]["name"] == "X"
        assert config["preprocessing"] == []
        assert config["feature_selection"] is None
        assert config["split"] is None


# ===========================================================================
# Workflow — __repr__
# ===========================================================================
class TestWorkflowRepr:
    def test_repr_with_data(self):
        wf = Workflow("iris.csv")
        r = repr(wf)
        assert "iris.csv" in r

    def test_repr_with_model(self):
        wf = Workflow().train("NaiveBayes")
        r = repr(wf)
        assert "NaiveBayes" in r

    def test_repr_with_preprocessing(self):
        wf = Workflow().impute().standardize()
        r = repr(wf)
        assert "Preprocessing" in r
        assert "2 steps" in r

    def test_repr_with_feature_selection(self):
        wf = Workflow().select_features(k=10)
        r = repr(wf)
        assert "Feature Selection" in r
        assert "SelectKBestSelector" in r


# ===========================================================================
# Workflow — get_parameter_schema
# ===========================================================================
class TestWorkflowSchema:
    def test_get_parameter_schema(self):
        schema = Workflow.get_parameter_schema()
        assert isinstance(schema, dict)
        assert "data" in schema
        assert "target" in schema


# ===========================================================================
# Workflow — validation errors
# ===========================================================================
class TestWorkflowValidation:
    def test_run_no_data(self):
        wf = Workflow().train("NaiveBayesClassifier")
        with pytest.raises(ValueError, match="No data provided"):
            wf.run()

    def test_run_no_model(self):
        X, y = _make_classification_data()
        wf = Workflow(X, target=y)
        with pytest.raises(ValueError, match="No model specified"):
            wf.run()


# ===========================================================================
# Workflow — run (integration tests with real algorithms)
# ===========================================================================
class TestWorkflowRun:
    def test_run_with_array(self):
        """Full pipeline: numpy array -> NaiveBayesClassifier -> holdout evaluation."""
        X, y = _make_classification_data(n_samples=100, n_features=4, seed=42)

        result = (
            Workflow(X, target=y)
            .train("NaiveBayesClassifier")
            .evaluate()
            .run()
        )

        assert isinstance(result, WorkflowResult)
        assert result.model is not None
        assert len(result.metrics) > 0
        assert result.predictions is not None
        assert result.metadata["algorithm"] == "NaiveBayesClassifier"

    def test_run_cross_validate(self):
        """Cross-validation should produce cv_* metrics and cv_results."""
        X, y = _make_classification_data(n_samples=100, n_features=4, seed=42)

        result = (
            Workflow(X, target=y)
            .train("NaiveBayesClassifier")
            .cross_validate(cv=3)
            .run()
        )

        assert isinstance(result, WorkflowResult)
        assert result.model is not None
        assert result.cv_results is not None
        # CV metrics should have mean and std keys
        metric_keys = list(result.metrics.keys())
        assert any("mean" in k for k in metric_keys)
        assert any("std" in k for k in metric_keys)

    def test_run_with_preprocessing(self):
        """Pipeline with standardize preprocessing."""
        X, y = _make_classification_data(n_samples=100, n_features=4, seed=42)

        result = (
            Workflow(X, target=y)
            .standardize()
            .train("NaiveBayesClassifier")
            .evaluate()
            .run()
        )

        assert isinstance(result, WorkflowResult)
        assert result.model is not None
        assert len(result.metrics) > 0

    def test_run_full_pipeline(self):
        """Full pipeline: impute + standardize + train + evaluate."""
        X, y = _make_classification_data(n_samples=120, n_features=4, seed=7)

        result = (
            Workflow(X, target=y)
            .impute(strategy="mean")
            .standardize()
            .train("NaiveBayesClassifier")
            .evaluate()
            .run()
        )

        assert isinstance(result, WorkflowResult)
        assert result.model is not None
        assert len(result.metrics) > 0
        assert result.metadata["preprocessing"] is not None
        assert len(result.metadata["preprocessing"]) == 2

    def test_run_result_can_predict(self):
        """WorkflowResult from run() should support predict on new data."""
        X, y = _make_classification_data(n_samples=80, n_features=4, seed=99)

        result = (
            Workflow(X, target=y)
            .train("NaiveBayesClassifier")
            .evaluate()
            .run()
        )

        X_new = np.random.RandomState(0).randn(5, 4)
        preds = result.predict(X_new)
        assert len(preds) == 5

    def test_run_holdout_evaluation_method(self):
        """Verify metadata reports holdout evaluation method."""
        X, y = _make_classification_data()

        result = (
            Workflow(X, target=y)
            .train("NaiveBayesClassifier")
            .evaluate()
            .run()
        )

        assert result.metadata["evaluation_method"] == "holdout"

    def test_run_cv_evaluation_method(self):
        """Verify metadata reports cross_validate evaluation method."""
        X, y = _make_classification_data()

        result = (
            Workflow(X, target=y)
            .train("NaiveBayesClassifier")
            .cross_validate(cv=3)
            .run()
        )

        assert result.metadata["evaluation_method"] == "cross_validate"

    def test_handle_missing_alias_runs(self):
        """handle_missing() and impute() should both work in a full pipeline."""
        X, y = _make_classification_data()

        result = (
            Workflow(X, target=y)
            .handle_missing(strategy="mean")
            .train("NaiveBayesClassifier")
            .evaluate()
            .run()
        )

        assert isinstance(result, WorkflowResult)
        assert result.model is not None

    def test_cross_validate_fits_preprocessing_per_fold(self, monkeypatch):
        """Preprocessing should fit on each train fold, not once on all data."""
        from tuiml.hub import registry

        X, y = _make_classification_data(n_samples=12, n_features=4, seed=1)
        _TrackingPreprocessor.fit_sizes = []
        _TrackingModel.fit_sizes = []

        def fake_resolve(name, fallback_module=None):
            if name == "TrackingScaler":
                return _TrackingPreprocessor
            return None

        def fake_get(name):
            if name == "TrackingModel":
                return _TrackingModel
            raise KeyError(name)

        def fake_get_info(name):
            if name == "TrackingModel":
                return {"type": "classifier", "tags": []}
            raise KeyError(name)

        monkeypatch.setattr(Workflow, "_resolve_component", staticmethod(fake_resolve))
        monkeypatch.setattr(registry, "get", fake_get)
        monkeypatch.setattr(registry, "get_info", fake_get_info)

        result = (
            Workflow(X, target=y)
            .preprocess("TrackingScaler")
            .train("TrackingModel")
            .cross_validate(cv=3)
            .run()
        )

        assert isinstance(result, WorkflowResult)
        assert sorted(_TrackingPreprocessor.fit_sizes) == [8, 8, 8, 12]
        assert sorted(_TrackingModel.fit_sizes) == [8, 8, 8, 12]

    def test_result_predict_applies_fitted_pipeline(self, monkeypatch):
        """WorkflowResult.predict should apply fitted preprocessing before inference."""
        from tuiml.hub import registry

        X, y = _make_classification_data(n_samples=20, n_features=4, seed=2)
        _TrackingPreprocessor.fit_sizes = []
        _TrackingModel.fit_sizes = []

        def fake_resolve(name, fallback_module=None):
            if name == "TrackingScaler":
                return _TrackingPreprocessor
            return None

        def fake_get(name):
            if name == "TrackingModel":
                return _TrackingModel
            raise KeyError(name)

        def fake_get_info(name):
            if name == "TrackingModel":
                return {"type": "classifier", "tags": []}
            raise KeyError(name)

        monkeypatch.setattr(Workflow, "_resolve_component", staticmethod(fake_resolve))
        monkeypatch.setattr(registry, "get", fake_get)
        monkeypatch.setattr(registry, "get_info", fake_get_info)

        result = (
            Workflow(X, target=y)
            .preprocess("TrackingScaler")
            .train("TrackingModel")
            .evaluate()
            .run()
        )

        X_new = np.array([[10.0, 20.0, 30.0, 40.0]])
        preds = result.predict(X_new)

        assert preds.shape == (1,)
        np.testing.assert_allclose(
            result.model.last_predict_input_,
            X_new - np.mean(X, axis=0),
        )
