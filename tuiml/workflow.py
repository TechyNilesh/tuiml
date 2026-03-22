"""Complete ML Workflow - Fluent chainable API for building ML pipelines."""

from typing import Optional, List, Dict, Any, Union
import numpy as np
import pandas as pd

class WorkflowResult:
    """Result object returned by ``Workflow.run()``.

    Contains the trained model, evaluation metrics, predictions, and
    additional metadata from the workflow execution.

    Attributes
    ----------
    model : object
        The trained model instance.

    model_id : str or None
        Unique identifier for the model (if registered).

    metrics : dict
        Dictionary of computed evaluation metrics.

    cv_results : dict or None
        Cross-validation results (if CV was used).

    predictions : ndarray or None
        Test set predictions (if requested).

    probabilities : ndarray or None
        Prediction probabilities (if requested).

    feature_importance : dict or None
        Feature importance scores (if available).

    preprocessing_pipeline : object or None
        Fitted preprocessing pipeline.

    metadata : dict
        Additional workflow metadata.
    """

    def __init__(
        self,
        model=None,
        model_id: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        cv_results: Optional[Dict] = None,
        predictions: Optional[np.ndarray] = None,
        probabilities: Optional[np.ndarray] = None,
        feature_importance: Optional[Dict[str, float]] = None,
        preprocessing_pipeline: Optional[Any] = None,
        metadata: Optional[Dict] = None,
    ):
        self.model = model
        self.model_id = model_id
        self.metrics = metrics or {}
        self.cv_results = cv_results
        self.predictions = predictions
        self.probabilities = probabilities
        self.feature_importance = feature_importance
        self.preprocessing_pipeline = preprocessing_pipeline
        self.metadata = metadata or {}

    def __repr__(self):
        return f"WorkflowResult(metrics={self.metrics})"

    def _transform_for_inference(self, X):
        """Apply any fitted preprocessing/feature-selection pipeline."""
        X_transformed = np.asarray(X)
        pipeline = self.preprocessing_pipeline
        if not pipeline:
            return X_transformed

        steps = pipeline.get('preprocessing_steps', []) if isinstance(pipeline, dict) else pipeline
        selector = pipeline.get('feature_selector') if isinstance(pipeline, dict) else None

        for step in steps:
            if hasattr(step, 'fit_resample'):
                # Resamplers are training-only; applying them at inference would
                # change the number of rows and break prediction semantics.
                continue
            transformed = step.transform(X_transformed)
            if isinstance(transformed, tuple):
                X_transformed = transformed[0]
            else:
                X_transformed = transformed

        if selector is not None:
            X_transformed = selector.transform(X_transformed)

        return X_transformed

    def predict(self, X) -> np.ndarray:
        """Make predictions using the trained model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to predict on.

        Returns
        -------
        predictions : np.ndarray
            Predicted values or class labels.

        Raises
        ------
        RuntimeError
            If no trained model is available.
        """
        if self.model is None:
            raise RuntimeError("No trained model available. Run the workflow first.")
        X_transformed = self._transform_for_inference(X)
        return self.model.predict(X_transformed)

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities using the trained model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to predict on.

        Returns
        -------
        probabilities : np.ndarray of shape (n_samples, n_classes)
            Predicted class probabilities.

        Raises
        ------
        RuntimeError
            If no trained model is available.
        AttributeError
            If the model does not support probability predictions.
        """
        if self.model is None:
            raise RuntimeError("No trained model available. Run the workflow first.")
        if not hasattr(self.model, 'predict_proba'):
            raise AttributeError(
                f"{self.model.__class__.__name__} does not support predict_proba()."
            )
        X_transformed = self._transform_for_inference(X)
        return self.model.predict_proba(X_transformed)

    def save(self, path: str) -> None:
        """Save the entire workflow result (model + metrics + metadata) to disk.

        Parameters
        ----------
        path : str
            File path to save to (e.g., ``"result.joblib"``).
        """
        import joblib
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "WorkflowResult":
        """Load a saved workflow result from disk.

        Parameters
        ----------
        path : str
            Path to the saved result file.

        Returns
        -------
        WorkflowResult
            The loaded workflow result.
        """
        import joblib
        return joblib.load(path)

    def serve(self, port: int = 8000, host: str = "127.0.0.1", model_id: str = None):
        """Serve the trained model via a REST API.

        Parameters
        ----------
        port : int, default=8000
            Port to listen on.
        host : str, default="127.0.0.1"
            Host to bind the server to.
        model_id : str, optional
            Identifier for the model. Defaults to the algorithm name
            from metadata or ``"default"``.

        Returns
        -------
        dict
            Server info dict with url and endpoints (if background),
            or None (if blocking).

        Raises
        ------
        RuntimeError
            If no trained model is available.
        """
        if self.model is None:
            raise RuntimeError("No trained model available. Run the workflow first.")

        if model_id is None:
            model_id = self.metadata.get('algorithm', 'default')

        from tuiml.api import serve as api_serve
        return api_serve(self, host=host, port=port, model_id=model_id)

class Workflow:
    """Complete ML workflow with a fluent chainable API.

    The Workflow class provides a convenient way to build end-to-end 
    machine learning pipelines using method chaining. Each method returns 
    ``self``, allowing you to chain operations together.

    Overview
    --------
    A typical workflow consists of:

    1. Initialize with data and target
    2. Add preprocessing steps (impute, normalize, encode)
    3. Optionally add feature selection
    4. Configure the model
    5. Set evaluation method
    6. Execute with ``run()``

    Parameters
    ----------
    data : str, Dataset, DataFrame, or ndarray, optional
        Training data source:

        - ``"iris"`` — Built-in dataset name
        - ``"path/to/file.csv"`` — File path (csv, arff, parquet, json, excel, numpy)
        - ``Dataset`` — TuiML Dataset object
        - ``DataFrame`` — Pandas DataFrame
        - ``ndarray`` — Numpy array

    target : str, optional
        Target column name (when data is a file path or DataFrame).

    Attributes
    ----------
    _preprocessing_steps : list
        List of preprocessing step configurations.

    _feature_selection : dict or None
        Feature selection configuration.

    _split_config : dict or None
        Train/test split configuration.

    _model : str or None
        Algorithm name to train.

    _model_params : dict
        Model hyperparameters.

    _evaluation_config : dict
        Evaluation configuration (holdout or CV).

    Examples
    --------
    Basic classification workflow:

    >>> result = (
    ...     Workflow("iris.csv", target="class")
    ...     .impute(strategy="mean")
    ...     .normalize()
    ...     .train("RandomForestClassifier", n_trees=100)
    ...     .cross_validate(cv=10)
    ...     .run()
    ... )

    Complete workflow with feature selection:

    >>> result = (
    ...     Workflow("data.csv", target="label")
    ...     .impute(strategy="knn")
    ...     .standardize()
    ...     .encode_categorical(method="onehot")
    ...     .select_features(k=20)
    ...     .train("SVM", kernel="rbf", C=1.0)
    ...     .evaluate()
    ...     .run()
    ... )

    Export configuration:

    >>> workflow = Workflow("data.csv", target="class")
    >>> workflow.train("NaiveBayesClassifier")
    >>> config = workflow.to_config()
    """

    @classmethod
    def get_parameter_schema(cls) -> dict:
        """Return JSON Schema for parameters."""
        return {
            "data": {
                "type": ["string", "object", "array", "null"],
                "default": None,
                "description": "Training data: file path, DataFrame, or numpy array"
            },
            "target": {
                "type": ["string", "null"],
                "default": None,
                "description": "Target column name (for DataFrame/file data)"
            }
        }

    def __init__(self, data: Union[str, pd.DataFrame, np.ndarray] = None, target: Optional[str] = None):
        """Initialize workflow.

        Parameters
        ----------
        data : str, DataFrame, or ndarray, optional
            Path to data file, DataFrame, or numpy array.
            
        target : str, optional
            Target column name (if data is DataFrame/file).
        """
        self._data = data
        self._target = target
        self._preprocessing_steps = []
        self._feature_selection = None
        self._split_config = None
        self._model = None
        self._model_params = {}
        self._evaluation_config = {}

    # ===== Data Loading =====

    def load(self, data: Union[str, pd.DataFrame, np.ndarray]) -> "Workflow":
        """Load data.

        Parameters
        ----------
        data : str, DataFrame, or ndarray
            Path to file, DataFrame, or array.

        Returns
        -------
        self : Workflow
            Self for chaining.
        """
        self._data = data
        return self

    # ===== Preprocessing Methods =====

    def impute(self, strategy: str = "mean", **kwargs) -> "Workflow":
        """Add imputation step for missing values.

        Parameters
        ----------
        strategy : str, default="mean"
            Imputation strategy:
            
            - ``"mean"`` — Mean imputation
            - ``"median"`` — Median imputation
            - ``"mode"`` — Mode imputation
            - ``"knn"`` — K-nearest neighbors imputation
            
        **kwargs
            Additional parameters for the imputer.

        Returns
        -------
        self : Workflow
            Self for chaining.
        """
        if strategy == "knn":
            self._preprocessing_steps.append({"name": "KNNImputer", **kwargs})
        else:
            self._preprocessing_steps.append(
                {"name": "SimpleImputer", "strategy": strategy, **kwargs}
            )
        return self

    def normalize(self, method: str = "minmax") -> "Workflow":
        """Add normalization step.

        Parameters
        ----------
        method : str, default="minmax"
            Normalization method:
            
            - ``"minmax"`` — Scale features to [0, 1] range
            - ``"zscore"`` — Scale features to zero mean and unit variance

        Returns
        -------
        self : Workflow
            Self for chaining.
        """
        if method == "zscore":
            self._preprocessing_steps.append({"name": "StandardScaler"})
        else:
            self._preprocessing_steps.append({"name": "MinMaxScaler"})
        return self

    def standardize(self) -> "Workflow":
        """Add standardization step (z-score normalization).

        Returns
        -------
        self : Workflow
            Self for chaining.
        """
        self._preprocessing_steps.append({"name": "StandardScaler"})
        return self

    def encode_categorical(self, method: str = "onehot") -> "Workflow":
        """Encode categorical variables.

        Parameters
        ----------
        method : str, default="onehot"
            Encoding method:
            
            - ``"onehot"`` — Create binary columns for each category
            - ``"ordinal"`` — Map categories to integers

        Returns
        -------
        self : Workflow
            Self for chaining.
        """
        if method == "onehot":
            self._preprocessing_steps.append({"name": "OneHotEncoder"})
        else:
            self._preprocessing_steps.append({"name": "OrdinalEncoder"})
        return self

    def handle_missing(self, strategy: str = "mean") -> "Workflow":
        """Alias for ``impute()``.

        Parameters
        ----------
        strategy : str, default="mean"
            Imputation strategy.

        Returns
        -------
        self : Workflow
            Self for chaining.
        """
        return self.impute(strategy)

    def resample(self, method: str = "smote", **kwargs) -> "Workflow":
        """Add resampling for imbalanced datasets.

        Parameters
        ----------
        method : str, default="smote"
            Resampling method:
            
            - ``"smote"`` — Synthetic Minority Over-sampling Technique
            - ``"adasyn"`` — Adaptive Synthetic sampling
            - ``"random_over"`` — Random over-sampling
            - ``"random_under"`` — Random under-sampling
            
        **kwargs
            Additional parameters for the resampler.

        Returns
        -------
        self : Workflow
            Self for chaining.
        """
        method_map = {
            "smote": "SMOTESampler",
            "adasyn": "ADASYNSampler",
            "random_over": "RandomOverSampler",
            "random_under": "RandomUnderSampler",
        }
        self._preprocessing_steps.append(
            {"name": method_map.get(method, method.upper()), **kwargs}
        )
        return self

    def preprocess(self, name: str, **kwargs) -> "Workflow":
        """Add any preprocessing step by name.

        Parameters
        ----------
        name : str
            Preprocessor class name (e.g., ``"SimpleImputer"``, ``"MinMaxScaler"``).
            
        **kwargs
            Parameters passed to the preprocessor.

        Returns
        -------
        self : Workflow
            Self for chaining.
        """
        self._preprocessing_steps.append({"name": name, **kwargs})
        return self

    # ===== Feature Engineering Methods =====

    def select_features(
        self, k: int = 10, **kwargs
    ) -> "Workflow":
        """Add feature selection step.

        Parameters
        ----------
        k : int, default=10
            Number of top features to select.
            
        **kwargs
            Additional parameters for the selector (e.g., ``score_func``).

        Returns
        -------
        self : Workflow
            Self for chaining.
        """
        self._feature_selection = {
            "name": "SelectKBestSelector",
            "k": k,
            **kwargs,
        }
        return self

    def pca(self, n_components: Union[int, float] = 0.95) -> "Workflow":
        """Add PCAExtractor dimensionality reduction.

        Parameters
        ----------
        n_components : int or float, default=0.95
            Number of components to keep. If float, it represents the
            proportion of variance to be explained.

        Returns
        -------
        self : Workflow
            Self for chaining.
        """
        self._feature_selection = {"name": "PCAExtractor", "n_components": n_components}
        return self

    # ===== Data Splitting =====

    def split(
        self,
        test_size: float = 0.2,
        stratify: bool = True,
        random_state: Optional[int] = 42,
    ) -> "Workflow":
        """Configure train/test split.

        Parameters
        ----------
        test_size : float, default=0.2
            Proportion of data reserved for testing.
            
        stratify : bool, default=True
            Whether to maintain class distribution in the split.
            
        random_state : int or None, default=42
            Random seed for reproducibility.

        Returns
        -------
        self : Workflow
            Self for chaining.
        """
        self._split_config = {
            "test_size": test_size,
            "stratify": stratify,
            "random_state": random_state,
        }
        return self

    # ===== Model Training =====

    def train(self, algorithm: str, **params) -> "Workflow":
        """Set the model algorithm to train.

        Parameters
        ----------
        algorithm : str
            Algorithm class name (e.g., ``"RandomForest"``, ``"SVM"``).
            
        **params
            Model hyperparameters.

        Returns
        -------
        self : Workflow
            Self for chaining.
        """
        self._model = algorithm
        self._model_params = params
        return self

    def model(self, algorithm: str, **params) -> "Workflow":
        """Alias for ``train()``.

        Parameters
        ----------
        algorithm : str
            Algorithm class name.
            
        **params
            Model hyperparameters.

        Returns
        -------
        self : Workflow
            Self for chaining.
        """
        return self.train(algorithm, **params)

    # ===== Evaluation Methods =====

    def evaluate(
        self, metrics: Optional[List[str]] = None, test_size: float = 0.2
    ) -> "Workflow":
        """Configure holdout evaluation.

        Parameters
        ----------
        metrics : list of str, optional
            List of metrics to compute (e.g., ``["accuracy", "f1"]``).
            
        test_size : float, default=0.2
            Test set size if split was not configured.

        Returns
        -------
        self : Workflow
            Self for chaining.
        """
        self._evaluation_config = {
            "method": "holdout",
            "metrics": metrics,
            "test_size": test_size,
        }
        return self

    def cross_validate(
        self, cv: int = 10, metrics: Optional[List[str]] = None
    ) -> "Workflow":
        """Configure cross-validation evaluation.

        Parameters
        ----------
        cv : int, default=10
            Number of cross-validation folds.
            
        metrics : list of str, optional
            List of metrics to compute.

        Returns
        -------
        self : Workflow
            Self for chaining.
        """
        self._evaluation_config = {
            "method": "cross_validate",
            "cv": cv,
            "metrics": metrics,
        }
        return self

    # ===== Execution =====

    def run(self) -> WorkflowResult:
        """Execute the complete workflow and return results.

        Returns
        -------
        WorkflowResult
            Object containing the trained model, metrics, and metadata.

        Raises
        ------
        ValueError
            If workflow is incomplete (missing data or model).
        """
        if self._data is None:
            raise ValueError("No data provided. Use load() or pass data in constructor.")

        if self._model is None:
            raise ValueError("No model specified. Use train() to set a model.")

        # Execute workflow directly
        return self._execute()

    @staticmethod
    def _resolve_component(name: str, fallback_module: str = None):
        """Resolve a component class by name via hub registry, with module fallback.

        Parameters
        ----------
        name : str
            Component class name (e.g., 'MinMaxScaler', 'SelectKBestSelector').
        fallback_module : str, optional
            Module path to try if not found in hub registry
            ('preprocessing' or 'features').

        Returns
        -------
        cls : type or None
            The component class, or None if not found.
        """
        from tuiml.hub import registry

        # 1. Try hub registry first (covers built-in + community/custom components)
        try:
            return registry.get(name)
        except KeyError:
            pass

        # 2. Fallback to module-level import for backward compatibility
        if fallback_module == 'preprocessing':
            try:
                from tuiml import preprocessing
                cls = getattr(preprocessing, name, None)
                if cls:
                    return cls
            except ImportError:
                pass
        elif fallback_module == 'features':
            try:
                from tuiml.features import selection, extraction, generation
                for module in (selection, extraction, generation):
                    cls = getattr(module, name, None)
                    if cls:
                        return cls
            except ImportError:
                pass

        return None

    @staticmethod
    def _normalize_step_config(step):
        """Normalize a preprocessing step config into name + params."""
        if isinstance(step, str):
            return step, {}
        if isinstance(step, dict):
            name = step.get('name')
            params = {k: v for k, v in step.items() if k not in ('name', 'params')}
            params.update(step.get('params', {}))
            return name, params
        return None, {}

    def _fit_preprocessing_pipeline(self, X, y=None):
        """Fit preprocessing steps on the provided data only."""
        X_current, y_current = X, y
        fitted_steps = []

        for step in self._preprocessing_steps:
            name, params = self._normalize_step_config(step)
            if not name:
                continue

            preprocessor_cls = self._resolve_component(name, 'preprocessing')
            if preprocessor_cls is None:
                raise ValueError(
                    f"Preprocessor '{name}' not found. Check the class name "
                    f"or ensure it is registered in the hub."
                )

            preprocessor = preprocessor_cls(**params)
            if hasattr(preprocessor, 'fit_resample') and y_current is not None:
                X_current, y_current = preprocessor.fit_resample(X_current, y_current)
            else:
                transformed = preprocessor.fit_transform(X_current, y_current)
                if isinstance(transformed, tuple):
                    X_current, y_current = transformed
                else:
                    X_current = transformed
            fitted_steps.append(preprocessor)

        return X_current, y_current, fitted_steps

    @staticmethod
    def _transform_with_preprocessing_pipeline(X, fitted_steps):
        """Transform inference/validation data with already-fitted steps."""
        X_current = X

        for step in fitted_steps:
            if hasattr(step, 'fit_resample'):
                continue
            transformed = step.transform(X_current)
            if isinstance(transformed, tuple):
                X_current = transformed[0]
            else:
                X_current = transformed

        return X_current

    def _fit_feature_selector(self, X, y=None):
        """Fit feature selection on the provided data only."""
        if not self._feature_selection:
            return X, None

        fs_name = self._feature_selection.get('name')
        fs_params = {
            k: v for k, v in self._feature_selection.items() if k != 'name'
        }
        selector_cls = self._resolve_component(fs_name, 'features')
        if selector_cls is None:
            raise ValueError(
                f"Feature selector '{fs_name}' not found. Check the class "
                f"name or ensure it is registered in the hub."
            )

        selector = selector_cls(**fs_params)
        if hasattr(selector, 'fit_transform'):
            X_selected = selector.fit_transform(X, y)
        else:
            selector.fit(X, y)
            X_selected = selector.transform(X)

        return X_selected, selector

    @staticmethod
    def _transform_with_feature_selector(X, selector):
        """Transform data with an already-fitted feature selector."""
        if selector is None:
            return X
        return selector.transform(X)

    def _fit_pipeline(self, X, y=None):
        """Fit preprocessing + feature selection on a dataset split."""
        X_processed, y_processed, fitted_steps = self._fit_preprocessing_pipeline(X, y)
        X_processed, selector = self._fit_feature_selector(X_processed, y_processed)
        pipeline = {
            'preprocessing_steps': fitted_steps,
            'feature_selector': selector,
        }
        return X_processed, y_processed, pipeline

    def _transform_pipeline(self, X, pipeline):
        """Apply a fitted pipeline to validation/test/inference data."""
        if not pipeline:
            return X
        X_processed = self._transform_with_preprocessing_pipeline(
            X, pipeline.get('preprocessing_steps', [])
        )
        return self._transform_with_feature_selector(
            X_processed, pipeline.get('feature_selector')
        )

    def _execute(self) -> WorkflowResult:
        """Internal execution method - performs actual training."""
        import numpy as np
        import os
        from tuiml.datasets import load_dataset, load
        from tuiml.hub import registry
        import tuiml.algorithms  # noqa: F401 - trigger registration
        import tuiml.preprocessing  # noqa: F401 - trigger registration
        import tuiml.features  # noqa: F401 - trigger registration

        # 1. Load data — use TuiML loaders for all file formats
        from tuiml.datasets.loaders.arff import Dataset as WNDataset

        if isinstance(self._data, WNDataset):
            # Already a Dataset object — use directly
            X, y = self._data.X, self._data.y

        elif isinstance(self._data, str):
            # String: try file path first (uses auto-detect loader for csv,
            # arff, parquet, json, excel, numpy), then built-in name
            if os.path.exists(self._data):
                dataset = load(self._data)
            else:
                dataset = load_dataset(self._data)
            X, y = dataset.X, dataset.y

        elif isinstance(self._data, pd.DataFrame):
            # DataFrame — use from_pandas to get a proper Dataset
            from tuiml.datasets import from_pandas
            dataset = from_pandas(
                self._data,
                target_column=self._target if isinstance(self._target, str) else None,
            )
            X = dataset.X
            y = dataset.y
            # If target was passed as a separate array, use that instead
            if isinstance(self._target, (np.ndarray, pd.Series)):
                y = np.asarray(self._target)

        else:
            # ndarray or array-like
            X = self._data if isinstance(self._data, np.ndarray) else np.asarray(self._data)
            y = np.asarray(self._target) if self._target is not None else None

        # 3. Determine split config (merge evaluate's test_size if split not configured)
        from tuiml.evaluation.splitting import train_test_split
        split_config = self._split_config or {}
        eval_config = self._evaluation_config or {}

        test_size = split_config.get(
            'test_size', eval_config.get('test_size', 0.2)
        )
        random_state = split_config.get('random_state', 42)
        do_stratify = split_config.get('stratify', True)
        stratify_arr = y if (do_stratify and y is not None) else None

        use_cv = eval_config.get('method') == 'cross_validate'

        # 4. Resolve model class
        try:
            model_cls = registry.get(self._model)
        except KeyError:
            raise ValueError(f"Unknown algorithm: {self._model}")

        model_params = self._model_params or {}
        _user_metrics = eval_config.get('metrics')

        # Dynamically load metrics from registry
        from tuiml.evaluation import metrics as metrics_module

        def _get_metric_func(metric_name):
            """Look up a metric function by name from the metrics module.

            Parameters
            ----------
            metric_name : str
                Exact function name (e.g., ``'accuracy_score'``,
                ``'f1_score'``, ``'mean_squared_error'``).

            Returns
            -------
            callable or None
                The metric function, or None if not found.
            """
            return getattr(metrics_module, metric_name, None)

        # Detect algorithm category from registry tags/type
        is_clustering = False
        is_anomaly = False
        is_timeseries = False
        algo_tags = []
        algo_info = {}
        try:
            algo_info = registry.get_info(self._model)
            algo_tags = algo_info.get('tags', [])
            is_clustering = algo_info.get('type') in ('clusterer', 'clustering')
            is_anomaly = 'anomaly-detection' in algo_tags
            is_timeseries = 'timeseries' in algo_tags
        except (KeyError, Exception):
            pass

        # Auto-resolve metrics based on algorithm type when not specified
        if _user_metrics:
            requested_metrics = _user_metrics
        elif is_clustering:
            requested_metrics = ['silhouette_score', 'calinski_harabasz_score']
        elif is_anomaly:
            requested_metrics = []
        elif is_timeseries:
            requested_metrics = ['r2_score', 'root_mean_squared_error', 'mean_absolute_error']
        else:
            algo_type = algo_info.get('type', 'classifier')
            if algo_type == 'regressor':
                requested_metrics = ['r2_score', 'mean_squared_error', 'mean_absolute_error']
            else:
                requested_metrics = ['accuracy_score', 'f1_score']

        metrics = {}
        cv_results = None
        y_pred = None
        preprocessing_pipeline = None

        if is_clustering:
            # ---- Clustering path ----
            # Clusterers use fit(X) only; metrics take (X, labels)
            X_model, _, preprocessing_pipeline = self._fit_pipeline(X, y)
            model = model_cls(**model_params)
            model.fit(X_model)
            labels = model.predict(X_model) if hasattr(model, 'predict') else model.labels_

            # Clustering metrics: silhouette_score(X, labels), etc.
            for metric_name in requested_metrics:
                metric_func = _get_metric_func(metric_name)
                if metric_func is not None:
                    try:
                        metrics[metric_name] = float(metric_func(X_model, labels))
                    except Exception as e:
                        metrics[f'{metric_name}_error'] = str(e)

        elif is_anomaly:
            # ---- Anomaly detection path ----
            # Anomaly detectors use fit(X) only (unsupervised).
            # predict(X) returns -1 (anomaly) / 1 (normal).
            # No train/test split — fit on all data and score it.
            X_model, _, preprocessing_pipeline = self._fit_pipeline(X, y)
            model = model_cls(**model_params)
            model.fit(X_model)
            predictions = model.predict(X_model)

            result_info = {
                'n_anomalies': int((predictions == -1).sum()),
                'n_normal': int((predictions == 1).sum()),
                'anomaly_ratio': float((predictions == -1).mean()),
            }

            # If decision_function is available, add score stats
            if hasattr(model, 'decision_function'):
                scores = model.decision_function(X_model)
                result_info['score_mean'] = float(np.mean(scores))
                result_info['score_std'] = float(np.std(scores))

            metrics = result_info

        elif is_timeseries:
            # ---- Timeseries path ----
            # Timeseries models use fit(y) with 1D series and
            # predict(steps=int) to forecast future values.
            # Use the first column of X as the time series if y is
            # absent or all-zeros; otherwise use y.
            series = y
            if series is None or (np.unique(series).size <= 1):
                # y is missing or constant (e.g. loaded as all-zeros) —
                # use the first feature column as the series.
                series = X[:, 0] if X.ndim == 2 else X

            # Train/test split for evaluation: hold out last portion
            n = len(series)
            split_idx = max(1, int(n * (1.0 - test_size)))
            train_series = series[:split_idx]
            test_series = series[split_idx:]

            model = model_cls(**model_params)
            model.fit(train_series)

            forecast_steps = len(test_series)
            if forecast_steps > 0:
                y_pred = model.predict(forecast_steps)

                for metric_name in requested_metrics:
                    metric_func = _get_metric_func(metric_name)
                    if metric_func is not None:
                        try:
                            metrics[metric_name] = float(
                                metric_func(test_series[:len(y_pred)], y_pred)
                            )
                        except Exception as e:
                            metrics[f'{metric_name}_error'] = str(e)

            # Re-fit on full series so the returned model is maximally useful
            model = model_cls(**model_params)
            model.fit(series)

        elif use_cv:
            # ---- Cross-validation path ----
            cv_folds = eval_config.get('cv', 10)
            from tuiml.evaluation.splitting import KFold

            kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            cv_scores = {
                m: [] for m in requested_metrics
                if _get_metric_func(m) is not None
            }

            for train_idx, val_idx in kfold.split(X, y):
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]

                X_tr, y_tr, fold_pipeline = self._fit_pipeline(X_tr, y_tr)
                X_val = self._transform_pipeline(X_val, fold_pipeline)

                fold_model = model_cls(**model_params)
                fold_model.fit(X_tr, y_tr)
                fold_pred = fold_model.predict(X_val)

                for metric_name in cv_scores:
                    metric_func = _get_metric_func(metric_name)
                    try:
                        score = metric_func(y_val, fold_pred)
                        cv_scores[metric_name].append(score)
                    except Exception:
                        pass

            for metric_name, scores in cv_scores.items():
                if scores:
                    metrics[f'cv_{metric_name}_mean'] = float(np.mean(scores))
                    metrics[f'cv_{metric_name}_std'] = float(np.std(scores))

            cv_results = {'scores': cv_scores}

            # Train final model on ALL data so the returned model is maximally useful
            X_final, y_final, preprocessing_pipeline = self._fit_pipeline(X, y)
            model = model_cls(**model_params)
            model.fit(X_final, y_final)
        else:
            # ---- Holdout path ----
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state,
                stratify=stratify_arr,
            )

            X_train, y_train, holdout_pipeline = self._fit_pipeline(X_train, y_train)
            X_test = self._transform_pipeline(X_test, holdout_pipeline)

            model = model_cls(**model_params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            for metric_name in requested_metrics:
                metric_func = _get_metric_func(metric_name)
                if metric_func is not None:
                    try:
                        metrics[metric_name] = float(metric_func(y_test, y_pred))
                    except Exception as e:
                        metrics[f'{metric_name}_error'] = str(e)

            X_final, y_final, preprocessing_pipeline = self._fit_pipeline(X, y)
            model = model_cls(**model_params)
            model.fit(X_final, y_final)

        # 5. Return results
        return WorkflowResult(
            model=model,
            metrics=metrics,
            predictions=y_pred,
            cv_results=cv_results,
            preprocessing_pipeline=preprocessing_pipeline,
            metadata={
                'algorithm': self._model,
                'preprocessing': self._preprocessing_steps,
                'feature_selection': self._feature_selection,
                'evaluation_method': 'cross_validate' if use_cv else 'holdout',
            }
        )

    def to_config(self) -> Dict[str, Any]:
        """Export workflow as a configuration dictionary.

        Returns
        -------
        dict
            Configuration dictionary containing data, preprocessing, 
            split, model, and evaluation settings.
        """
        return {
            "data": {
                "source": self._data if isinstance(self._data, str) else None,
                "target": self._target,
            },
            "preprocessing": self._preprocessing_steps,
            "feature_selection": self._feature_selection,
            "split": self._split_config,
            "model": {"name": self._model, "params": self._model_params},
            "evaluation": self._evaluation_config,
        }

    def __repr__(self):
        steps = []
        steps.append(f"Data: {self._data}")
        if self._preprocessing_steps:
            steps.append(f"Preprocessing: {len(self._preprocessing_steps)} steps")
        if self._feature_selection:
            steps.append(f"Feature Selection: {self._feature_selection['name']}")
        if self._model:
            steps.append(f"Model: {self._model}")
        return f"Workflow({' → '.join(steps)})"
