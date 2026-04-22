"""
Core workflow tools for LLM integration.

Provides high-level task-oriented tools that LLMs can use
to perform complete ML workflows.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import os
import uuid
import tempfile

# Persistent TuiML state directory (survives MCP server restarts, unlike /tmp).
_TUIML_HOME = os.path.join(os.path.expanduser('~'), '.tuiml')
_MODELS_DIR = os.path.join(_TUIML_HOME, 'models')
_UPLOADS_DIR = os.path.join(_TUIML_HOME, 'uploads')
os.makedirs(_MODELS_DIR, exist_ok=True)
os.makedirs(_UPLOADS_DIR, exist_ok=True)

# Maps model_id -> file path on disk
_MODEL_INDEX: Dict[str, str] = {}

# Maps dataset_id (user-provided name or auto-generated) -> file path on disk.
# Scanned at import time so uploads persist across MCP server restarts.
_DATASET_INDEX: Dict[str, str] = {}
for _f in os.listdir(_UPLOADS_DIR):
    _full = os.path.join(_UPLOADS_DIR, _f)
    if os.path.isfile(_full):
        _DATASET_INDEX[os.path.splitext(_f)[0]] = _full

# Serving state: tracks running API servers
_SERVING_SERVERS: Dict[str, Dict[str, Any]] = {}  # server_id -> {thread, server, port, ...}


def _save_model_to_disk(model, model_id: str, save_path: str = None) -> str:
    """Save model to disk and return the file path."""
    import tuiml

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        tuiml.save(model, save_path)
        return save_path
    else:
        path = os.path.join(_MODELS_DIR, f'{model_id}.joblib')
        tuiml.save(model, path)
        return path


def _load_model_from_disk(model_id: str = None, model_path: str = None):
    """Load model from disk by model_id or explicit path."""
    import tuiml

    if model_id and model_id in _MODEL_INDEX:
        return tuiml.load(_MODEL_INDEX[model_id])
    elif model_path and os.path.exists(model_path):
        return tuiml.load(model_path)
    return None

def _load_data(data_source: str):
    """Load data from an uploaded dataset_id, file path, or built-in dataset name.

    Resolution order:
      1. Uploaded dataset_id (registered via tuiml_upload_data)
      2. Existing file path on disk
      3. Built-in dataset name (iris, diabetes, ...)
    """
    from tuiml.datasets import load, load_dataset

    # 1. Uploaded dataset registered by name
    if data_source in _DATASET_INDEX:
        path = _DATASET_INDEX[data_source]
        if os.path.exists(path):
            return load(path)
        # Stale entry — drop and fall through
        _DATASET_INDEX.pop(data_source, None)

    # 2. File path on disk
    if os.path.exists(data_source):
        return load(data_source)

    # 3. Built-in dataset name
    return load_dataset(data_source)

# =============================================================================
# Tool Schemas (JSON Schema format for MCP)
# =============================================================================

# Output Schemas for MCP Tools
OUTPUT_SCHEMAS = {
    "tuiml_train": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["success", "error"]},
            "model_id": {
                "type": "string",
                "description": "Model ID - use with tuiml_predict and tuiml_evaluate"
            },
            "model_path": {
                "type": "string",
                "description": "File path where the model is saved on disk"
            },
            "metrics": {
                "type": "object",
                "description": "Performance metrics (accuracy, f1, etc.)"
            },
            "cv_results": {
                "type": "object",
                "description": "Cross-validation fold results"
            },
            "model_class": {
                "type": "string",
                "description": "Name of the trained model class"
            },
            "metadata": {"type": "object"},
            "error": {"type": "string"}
        },
        "required": ["status"]
    },
    "tuiml_predict": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["success", "error"]},
            "num_predictions": {"type": "integer"},
            "predictions_preview": {
                "type": "array",
                "description": "First 10 predictions"
            },
            "output_path": {"type": "string"},
            "error": {"type": "string"}
        },
        "required": ["status"]
    },
    "tuiml_evaluate": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["success", "error"]},
            "metrics": {
                "type": "object",
                "description": "Evaluation metrics"
            },
            "error": {"type": "string"}
        },
        "required": ["status"]
    },
    "tuiml_experiment": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["success", "error"]},
            "summary": {"type": "string"},
            "results": {
                "type": "object",
                "description": "Results by dataset and model"
            },
            "algorithms": {"type": "array", "items": {"type": "string"}},
            "datasets": {"type": "array", "items": {"type": "string"}},
            "cv_folds": {"type": "integer"},
            "error": {"type": "string"},
            "suggested_metrics": {"type": "array", "items": {"type": "string"}},
            "algorithm_types": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["status"]
    },
    "tuiml_upload_data": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["success", "error"]},
            "dataset_id": {
                "type": "string",
                "description": "Stable name to pass as `data` to tuiml_train / tuiml_predict / tuiml_evaluate"
            },
            "file_path": {"type": "string"},
            "rows": {"type": "integer"},
            "features": {"type": "integer"},
            "feature_names": {"type": "array", "items": {"type": "string"}},
            "message": {"type": "string"},
            "error": {"type": "string"}
        },
        "required": ["status"]
    },
    "tuiml_save_model": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["success", "error"]},
            "model_id": {"type": "string"},
            "source": {"type": "string"},
            "destination": {"type": "string"},
            "message": {"type": "string"},
            "error": {"type": "string"}
        },
        "required": ["status"]
    },
    "tuiml_serve_model": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["success", "error"]},
            "server_id": {"type": "string"},
            "model_id": {"type": "string"},
            "url": {"type": "string", "description": "Base URL of the serving API"},
            "endpoints": {"type": "object", "description": "Map of endpoint names to URLs"},
            "example_curl": {"type": "string"},
            "error": {"type": "string"}
        },
        "required": ["status"]
    },
    "tuiml_stop_server": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["success", "error"]},
            "message": {"type": "string"},
            "stopped": {"type": "array"},
            "error": {"type": "string"}
        },
        "required": ["status"]
    },
    "tuiml_server_status": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["success", "error"]},
            "count": {"type": "integer"},
            "servers": {"type": "array"},
            "error": {"type": "string"}
        },
        "required": ["status"]
    },
    "tuiml_list": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["success", "error"]},
            "total": {"type": "integer", "description": "Total number of components"},
            "count": {"type": "integer", "description": "Number of components returned"},
            "limit": {"type": "integer"},
            "offset": {"type": "integer"},
            "has_more": {"type": "boolean"},
            "components": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "category": {"type": "string"}
                    }
                }
            },
            "error": {"type": "string"}
        },
        "required": ["status"]
    },
    "tuiml_describe": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["success", "error"]},
            "type": {"type": "string"},
            "name": {"type": "string"},
            "description": {"type": "string"},
            "parameters": {"type": "object"},
            "error": {"type": "string"}
        },
        "required": ["status"]
    },
    "tuiml_search": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["success", "error"]},
            "query": {"type": "string"},
            "count": {"type": "integer"},
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "category": {"type": "string"}
                    }
                }
            },
            "error": {"type": "string"}
        },
        "required": ["status"]
    },
    "tuiml_plot": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["success", "error"]},
            "plot_type": {"type": "string"},
            "description": {"type": "string"},
            "error": {"type": "string"}
        },
        "required": ["status"]
    },
    "tuiml_data_profile": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["success", "error"]},
            "name": {"type": "string"},
            "shape": {"type": "array", "items": {"type": "integer"}},
            "n_samples": {"type": "integer"},
            "n_features": {"type": "integer"},
            "feature_names": {"type": "array", "items": {"type": "string"}},
            "dtypes": {"type": "object"},
            "missing_values": {"type": "object"},
            "numeric_stats": {"type": "object"},
            "class_distribution": {"type": "object"},
            "target_column": {"type": "string"},
            "error": {"type": "string"}
        },
        "required": ["status"]
    },
    "tuiml_generate_data": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["success", "error"]},
            "generator": {"type": "string"},
            "file_path": {"type": "string"},
            "shape": {"type": "array", "items": {"type": "integer"}},
            "feature_names": {"type": "array", "items": {"type": "string"}},
            "target_names": {"type": "array", "items": {"type": "string"}},
            "preview": {"type": "object"},
            "error": {"type": "string"}
        },
        "required": ["status"]
    },
    "tuiml_preprocess": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["success", "error"]},
            "file_path": {"type": "string"},
            "original_shape": {"type": "array", "items": {"type": "integer"}},
            "new_shape": {"type": "array", "items": {"type": "integer"}},
            "steps_applied": {"type": "array", "items": {"type": "string"}},
            "error": {"type": "string"}
        },
        "required": ["status"]
    },
    "tuiml_select_features": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["success", "error"]},
            "method": {"type": "string"},
            "n_original": {"type": "integer"},
            "n_selected": {"type": "integer"},
            "selected_features": {"type": "array", "items": {"type": "string"}},
            "scores": {"type": "object"},
            "file_path": {"type": "string"},
            "error": {"type": "string"}
        },
        "required": ["status"]
    },
    "tuiml_statistical_test": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["success", "error"]},
            "test": {"type": "string"},
            "statistic": {"type": "number"},
            "p_value": {"type": "number"},
            "significant": {"type": "boolean"},
            "details": {"type": "object"},
            "error": {"type": "string"}
        },
        "required": ["status"]
    },
    "tuiml_tune": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["success", "error"]},
            "method": {"type": "string"},
            "best_params": {"type": "object"},
            "best_score": {"type": "number"},
            "cv_results": {"type": "object"},
            "model_id": {"type": "string"},
            "model_path": {"type": "string"},
            "error": {"type": "string"}
        },
        "required": ["status"]
    },
    "tuiml_read_data": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["success", "error"]},
            "name": {"type": "string"},
            "shape": {"type": "array", "items": {"type": "integer"}},
            "columns": {"type": "array", "items": {"type": "string"}},
            "n_rows_returned": {"type": "integer"},
            "rows": {"type": "array", "items": {"type": "object"}},
            "error": {"type": "string"}
        },
        "required": ["status"]
    }
}

# Component tool output schema (generic for all component tools)
COMPONENT_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["success", "error"]},
        "result": {"type": "string", "description": "String representation of the component"},
        "type": {"type": "string", "description": "Component class name"},
        "error": {"type": "string"}
    },
    "required": ["status"]
}

WORKFLOW_TOOLS = {
    "tuiml_train": {
        "name": "tuiml_train",
        "description": (
            "Train a machine learning model with evaluation. Two evaluation modes:\n"
            "1. Holdout (default): splits data into train/test sets using test_size. "
            "Returns metrics on the test set and predictions.\n"
            "2. Cross-validation: set cv=5 or cv=10 for k-fold CV. "
            "Returns mean/std metrics across folds.\n"
            "If neither cv nor test_size is provided, defaults to holdout with test_size=0.2.\n"
            "Supports classifiers, regressors, and clusterers."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "algorithm": {
                    "type": "string",
                    "description": (
                        "Algorithm class name. Examples:\n"
                        "- Classifiers: 'RandomForestClassifier', 'SVM', 'NaiveBayesClassifier', 'C45TreeClassifier'\n"
                        "- Regressors: 'LinearRegression', 'M5ModelTreeRegressor'\n"
                        "- Clusterers: 'KMeansClusterer', 'GaussianMixtureClusterer', 'DBSCANClusterer'"
                    )
                },
                "data": {
                    "type": "string",
                    "description": "Data file path or built-in dataset name (e.g., 'iris', 'wine')"
                },
                "target": {
                    "type": "string",
                    "description": "Target column (required for supervised, optional for clustering)"
                },
                "preprocessing": {
                    "type": "array",
                    "items": {
                        "oneOf": [
                            {"type": "string"},
                            {
                                "type": "object",
                                "properties": {"name": {"type": "string"}},
                                "required": ["name"],
                                "additionalProperties": True
                            }
                        ]
                    },
                    "description": (
                        "Preprocessing steps as names or objects with params.\n"
                        "Examples: ['SimpleImputer', 'StandardScaler'] or "
                        "[{'name': 'SimpleImputer', 'strategy': 'median'}, 'MinMaxScaler']"
                    )
                },
                "feature_selection": {
                    "oneOf": [
                        {"type": "string"},
                        {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                            },
                            "required": ["name"],
                            "additionalProperties": True
                        }
                    ],
                    "description": (
                        "Feature selection method. String name or object with params.\n"
                        "Examples: 'SelectKBestSelector' or {'name': 'SelectKBestSelector', 'k': 10}"
                    )
                },
                "cv": {
                    "type": "integer",
                    "description": (
                        "Number of cross-validation folds (e.g. 5 or 10). "
                        "OPTIONAL: if omitted, uses holdout train/test split instead. "
                        "Only used for supervised learning (ignored for clustering)."
                    )
                },
                "test_size": {
                    "type": "number",
                    "default": 0.2,
                    "description": (
                        "Proportion of data for the test set (0.0-1.0). "
                        "Used in holdout mode (when cv is NOT set). Default 0.2 (80/20 split). "
                        "Ignored when cv is set."
                    )
                },
                "metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Metrics to compute. Use exact function names. Must match algorithm type:\n"
                        "- Classification: ['accuracy_score', 'f1_score', 'precision_score', 'recall_score', 'roc_auc_score', 'balanced_accuracy_score', 'log_loss', 'matthews_corrcoef']\n"
                        "- Regression: ['r2_score', 'root_mean_squared_error', 'mean_absolute_error', 'mean_squared_error']\n"
                        "- Clustering: ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score']"
                    )
                },
                "preset": {
                    "type": "string",
                    "enum": ["minimal", "fast", "standard", "full", "imbalanced"],
                    "description": "Preprocessing preset"
                },
                "algorithm_params": {
                    "type": "object",
                    "description": "Algorithm hyperparameters (e.g., {'n_clusters': 3})"
                },
                "save_path": {
                    "type": "string",
                    "description": "Custom path to save the model file (optional). If omitted, saved to temp directory."
                }
            },
            "required": ["algorithm", "data"]
        }
    },

    "tuiml_predict": {
        "name": "tuiml_predict",
        "description": (
            "Make predictions using a trained model on new data. Supports supervised models, "
            "timeseries models (use 'steps' parameter), and anomaly detection models."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_id": {
                    "type": "string",
                    "description": "Model ID returned by tuiml_train (preferred)"
                },
                "model_path": {
                    "type": "string",
                    "description": "Path to saved model file (.pkl) (alternative to model_id)"
                },
                "data": {
                    "type": "string",
                    "description": "Path to data file for prediction"
                },
                "steps": {
                    "type": "integer",
                    "description": "Number of forecast steps (timeseries models only)"
                },
                "output_path": {
                    "type": "string",
                    "description": "Path to save predictions (optional)"
                }
            },
            "required": []
        }
    },

    "tuiml_evaluate": {
        "name": "tuiml_evaluate",
        "description": "Evaluate a trained model on test data and compute metrics.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_id": {
                    "type": "string",
                    "description": "Model ID returned by tuiml_train (preferred)"
                },
                "model_path": {
                    "type": "string",
                    "description": "Path to saved model file (alternative to model_id)"
                },
                "data": {
                    "type": "string",
                    "description": "Path to test data file"
                },
                "target": {
                    "type": "string",
                    "description": "Target column name"
                },
                "metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Metrics to compute"
                }
            },
            "required": ["data", "target"]
        }
    },

    "tuiml_experiment": {
        "name": "tuiml_experiment",
        "description": "Compare multiple algorithms on one or more datasets with cross-validation and statistical tests. Supports supervised learning (classification, regression) and unsupervised learning (clustering). Pass a single dataset name or a list of dataset names to benchmark across multiple datasets.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "algorithms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of algorithm class names to compare (e.g., ['RandomForestClassifier', 'SVM'] for classification, ['KMeansClusterer', 'GaussianMixtureClusterer'] for clustering)"
                },
                "data": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}}
                    ],
                    "description": "Dataset name(s) or file path(s). Single string (e.g., 'iris') or list of names (e.g., ['iris', 'wine', 'breast_cancer']) to compare across multiple datasets."
                },
                "target": {
                    "type": "string",
                    "description": "Target column name (for supervised learning)"
                },
                "cv": {
                    "type": "integer",
                    "default": 10,
                    "description": "Number of CV folds (ignored for clustering)"
                },
                "metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Metrics to compute. Use exact function names. IMPORTANT: Must match algorithm type:\n"
                        "- Classification: ['accuracy_score', 'f1_score', 'precision_score', 'recall_score', 'roc_auc_score']\n"
                        "- Regression: ['r2_score', 'root_mean_squared_error', 'mean_absolute_error', 'mean_squared_error']\n"
                        "- Clustering: ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score']\n"
                        "If omitted, appropriate metrics are automatically selected based on algorithm type."
                    )
                }
            },
            "required": ["algorithms", "data", "target"]
        }
    },

    "tuiml_upload_data": {
        "name": "tuiml_upload_data",
        "description": (
            "Register a dataset for use with other TuiML tools. "
            "Provide either a file_path to an existing file on disk (preferred for large datasets), "
            "or content as raw text for small inline datasets. "
            "Supported formats: CSV, TSV, ARFF, Parquet, Excel (xlsx/xls), JSON, JSONL, NumPy (npy/npz). "
            "Returns a validated path for use with tuiml_train, tuiml_preprocess, etc."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": (
                        "Path to an existing dataset file on disk. "
                        "Supported: .csv, .tsv, .arff, .parquet, .pq, .xlsx, .xls, .json, .jsonl, .ndjson, .npy, .npz"
                    )
                },
                "content": {
                    "type": "string",
                    "description": "Raw text content for small inline datasets (use with 'format')"
                },
                "format": {
                    "type": "string",
                    "enum": ["csv", "tsv", "arff", "json", "jsonl"],
                    "default": "csv",
                    "description": "File format — only needed with 'content'; auto-detected from file_path extension"
                },
                "name": {
                    "type": "string",
                    "description": "Optional name for the dataset (without extension)"
                }
            },
            "required": []
        }
    },

    "tuiml_save_model": {
        "name": "tuiml_save_model",
        "description": "Copy a trained model to a custom path. Use this when the user wants to save or download a model to a specific location.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_id": {
                    "type": "string",
                    "description": "Model ID returned by tuiml_train"
                },
                "destination": {
                    "type": "string",
                    "description": "Destination file path (e.g., './my_model.joblib', '/home/user/models/rf.joblib')"
                }
            },
            "required": ["model_id", "destination"]
        }
    },

    "tuiml_serve_model": {
        "name": "tuiml_serve_model",
        "description": (
            "Start a REST API server to serve a trained model for predictions. "
            "Returns the URL with endpoints: POST /predict, POST /models/{id}/predict, "
            "GET /health, GET /models, GET /docs (Swagger UI)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_id": {
                    "type": "string",
                    "description": "Model ID returned by tuiml_train"
                },
                "model_path": {
                    "type": "string",
                    "description": "Path to model file (alternative to model_id)"
                },
                "port": {
                    "type": "integer",
                    "default": 8000,
                    "minimum": 1024,
                    "maximum": 65535,
                    "description": "Port to serve on (default: 8000)"
                },
                "host": {
                    "type": "string",
                    "default": "127.0.0.1",
                    "description": "Host to bind to (default: 127.0.0.1)"
                }
            },
            "required": []
        }
    },

    "tuiml_stop_server": {
        "name": "tuiml_stop_server",
        "description": "Stop a running model serving API server.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "server_id": {
                    "type": "string",
                    "description": "Server ID returned by tuiml_serve_model. If omitted, stops all servers."
                }
            },
            "required": []
        }
    },

    "tuiml_server_status": {
        "name": "tuiml_server_status",
        "description": "Get status of running model serving API servers.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },

    "tuiml_plot": {
        "name": "tuiml_plot",
        "description": (
            "Generate a visualization/plot for model analysis. Returns the plot as an "
            "inline image. Supported plot types: confusion_matrix, roc_curve, pr_curve, "
            "learning_curve, tree, feature_importance."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "plot_type": {
                    "type": "string",
                    "enum": [
                        "confusion_matrix",
                        "roc_curve",
                        "pr_curve",
                        "learning_curve",
                        "tree",
                        "feature_importance",
                        "cd_diagram",
                        "boxplot_comparison",
                        "heatmap",
                        "ranking_table"
                    ],
                    "description": (
                        "Type of plot to generate:\n"
                        "- confusion_matrix: Heatmap of predicted vs actual classes (requires model_id + data + target)\n"
                        "- roc_curve: ROC curve with AUC for binary classifiers (requires model_id + data + target)\n"
                        "- pr_curve: Precision-Recall curve with AP for binary classifiers (requires model_id + data + target)\n"
                        "- learning_curve: Training vs validation score over dataset sizes (requires algorithm + data + target)\n"
                        "- tree: Decision tree structure visualization (requires model_id)\n"
                        "- feature_importance: Bar chart of feature importances (requires model_id)\n"
                        "- cd_diagram: Critical difference diagram for algorithm comparison (requires experiment_results)\n"
                        "- boxplot_comparison: Box plot comparing algorithm scores (requires experiment_results)\n"
                        "- heatmap: Heatmap of algorithm scores across datasets (requires experiment_results)\n"
                        "- ranking_table: Ranking table of algorithms (requires experiment_results)"
                    )
                },
                "model_id": {
                    "type": "string",
                    "description": "Model ID from tuiml_train (required for most plot types)"
                },
                "model_path": {
                    "type": "string",
                    "description": "Path to saved model file (alternative to model_id)"
                },
                "data": {
                    "type": "string",
                    "description": "Data file path or built-in dataset name (required for confusion_matrix, roc_curve, pr_curve, learning_curve)"
                },
                "target": {
                    "type": "string",
                    "description": "Target column name (required for confusion_matrix, roc_curve, pr_curve, learning_curve)"
                },
                "algorithm": {
                    "type": "string",
                    "description": "Algorithm class name (required for learning_curve)"
                },
                "title": {
                    "type": "string",
                    "description": "Custom plot title (optional)"
                },
                "normalize": {
                    "type": "boolean",
                    "default": False,
                    "description": "Normalize confusion matrix to show percentages (confusion_matrix only)"
                },
                "experiment_results": {
                    "type": "object",
                    "description": "Algorithm CV scores for comparison plots: { 'AlgoName': [score1, score2, ...], ... }"
                }
            },
            "required": ["plot_type"]
        }
    },

    "tuiml_data_profile": {
        "name": "tuiml_data_profile",
        "description": (
            "Inspect a dataset before training — shape, dtypes, missing values, "
            "basic statistics, and class distribution. Works with file paths or "
            "built-in dataset names."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "data": {
                    "type": "string",
                    "description": "Data file path or built-in dataset name (e.g., 'iris', 'wine', '/path/to/data.csv')"
                },
                "target": {
                    "type": "string",
                    "description": "Target column name (optional, used for class distribution)"
                }
            },
            "required": ["data"]
        }
    },

    "tuiml_generate_data": {
        "name": "tuiml_generate_data",
        "description": (
            "Generate synthetic datasets for testing and demos. Supports classification "
            "(RandomRBF, Agrawal, LED, Hyperplane), regression (Friedman, MexicanHat, Sine), "
            "and clustering (Blobs, Moons, Circles, SwissRoll) generators."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "generator": {
                    "type": "string",
                    "enum": [
                        "RandomRBF", "Agrawal", "LED", "Hyperplane",
                        "Friedman", "MexicanHat", "Sine",
                        "Blobs", "Moons", "Circles", "SwissRoll"
                    ],
                    "description": "Generator class name"
                },
                "n_samples": {
                    "type": "integer",
                    "default": 100,
                    "description": "Number of samples to generate"
                },
                "n_features": {
                    "type": "integer",
                    "description": "Number of features (not all generators support this)"
                },
                "n_classes": {
                    "type": "integer",
                    "description": "Number of classes (classification generators only)"
                },
                "n_clusters": {
                    "type": "integer",
                    "description": "Number of clusters (clustering generators only)"
                },
                "noise": {
                    "type": "number",
                    "description": "Noise level (regression generators only)"
                },
                "random_state": {
                    "type": "integer",
                    "description": "Random seed for reproducibility"
                },
                "generator_params": {
                    "type": "object",
                    "description": "Additional generator-specific parameters"
                }
            },
            "required": ["generator"]
        }
    },

    "tuiml_preprocess": {
        "name": "tuiml_preprocess",
        "description": (
            "Apply preprocessing steps to a dataset and return the result as a new file. "
            "Supports any registered preprocessor (e.g., StandardScaler, MinMaxScaler, "
            "SimpleImputer, PCA). Steps can be strings or objects with parameters."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "data": {
                    "type": "string",
                    "description": "Data file path or built-in dataset name"
                },
                "target": {
                    "type": "string",
                    "description": "Target column name (excluded from preprocessing, re-appended to output)"
                },
                "steps": {
                    "type": "array",
                    "items": {
                        "oneOf": [
                            {"type": "string"},
                            {
                                "type": "object",
                                "properties": {"name": {"type": "string"}},
                                "required": ["name"],
                                "additionalProperties": True
                            }
                        ]
                    },
                    "description": (
                        "Preprocessing steps as names or objects with params.\n"
                        "Examples: ['StandardScaler', 'SimpleImputer'] or "
                        "[{'name': 'SimpleImputer', 'strategy': 'median'}, 'MinMaxScaler']"
                    )
                },
                "save_as": {
                    "type": "string",
                    "description": "Custom output file path (optional, defaults to temp file)"
                }
            },
            "required": ["data", "steps"]
        }
    },

    "tuiml_select_features": {
        "name": "tuiml_select_features",
        "description": (
            "Run feature selection on a dataset and return selected feature names/indices. "
            "Supports filter methods (SelectKBestSelector, SelectPercentileSelector, "
            "VarianceThresholdSelector, SelectFprSelector, SelectThresholdSelector), "
            "correlation-based (CFSSelector), and wrapper methods (WrapperSelector)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "data": {
                    "type": "string",
                    "description": "Data file path or built-in dataset name"
                },
                "target": {
                    "type": "string",
                    "description": "Target column name"
                },
                "method": {
                    "type": "string",
                    "enum": [
                        "SelectKBestSelector", "SelectPercentileSelector",
                        "VarianceThresholdSelector", "CFSSelector",
                        "WrapperSelector", "SelectFprSelector", "SelectThresholdSelector"
                    ],
                    "description": "Feature selection method"
                },
                "k": {
                    "type": "integer",
                    "description": "Number of top features to select (SelectKBestSelector)"
                },
                "threshold": {
                    "type": "number",
                    "description": "Threshold for VarianceThresholdSelector or SelectThresholdSelector"
                },
                "method_params": {
                    "type": "object",
                    "description": "Additional method-specific parameters"
                }
            },
            "required": ["data", "target", "method"]
        }
    },

    "tuiml_statistical_test": {
        "name": "tuiml_statistical_test",
        "description": (
            "Run statistical significance tests on experiment results (cross-validation scores). "
            "Supports Friedman test, Nemenyi post-hoc, Wilcoxon signed-rank, paired t-test, "
            "one-way ANOVA, Friedman aligned ranks, and Quade test."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "test": {
                    "type": "string",
                    "enum": [
                        "friedman", "nemenyi", "wilcoxon",
                        "paired_t", "anova", "friedman_aligned", "quade"
                    ],
                    "description": (
                        "Statistical test to run:\n"
                        "- friedman: Non-parametric test for 3+ algorithms\n"
                        "- nemenyi: Post-hoc pairwise test after Friedman\n"
                        "- wilcoxon: Non-parametric pairwise test (2 algorithms)\n"
                        "- paired_t: Parametric pairwise test (2 algorithms)\n"
                        "- anova: Parametric test for 3+ groups\n"
                        "- friedman_aligned: More powerful variant of Friedman\n"
                        "- quade: Non-parametric test accounting for dataset difficulty"
                    )
                },
                "results": {
                    "type": "object",
                    "description": "Algorithm CV scores: { 'AlgorithmName': [score1, score2, ...], ... }"
                },
                "significance_level": {
                    "type": "number",
                    "default": 0.05,
                    "description": "Significance level (alpha), default 0.05"
                },
                "higher_better": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether higher scores are better (default True)"
                }
            },
            "required": ["test", "results"]
        }
    },

    "tuiml_tune": {
        "name": "tuiml_tune",
        "description": (
            "Hyperparameter optimization for any algorithm. Supports grid search, "
            "random search, and Bayesian optimization. Returns best parameters, "
            "best score, and a trained model with optimal settings."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "algorithm": {
                    "type": "string",
                    "description": "Algorithm class name (e.g., 'RandomForestClassifier', 'SVM')"
                },
                "data": {
                    "type": "string",
                    "description": "Data file path or built-in dataset name"
                },
                "target": {
                    "type": "string",
                    "description": "Target column name"
                },
                "method": {
                    "type": "string",
                    "enum": ["grid", "random", "bayesian"],
                    "description": "Tuning method: 'grid' (exhaustive), 'random' (sampled), 'bayesian' (GP-based)"
                },
                "param_grid": {
                    "type": "object",
                    "description": (
                        "Parameter search space. For grid: {'param': [val1, val2]}. "
                        "For random/bayesian: {'param': [low, high, 'int']} or {'param': [val1, val2]}."
                    )
                },
                "cv": {
                    "type": "integer",
                    "default": 5,
                    "description": "Number of cross-validation folds"
                },
                "scoring": {
                    "type": "string",
                    "description": "Scoring metric (e.g., 'accuracy', 'r2', 'neg_mse')"
                },
                "n_iter": {
                    "type": "integer",
                    "default": 10,
                    "description": "Number of iterations for random search"
                },
                "n_iterations": {
                    "type": "integer",
                    "default": 50,
                    "description": "Number of iterations for Bayesian search"
                },
                "random_state": {
                    "type": "integer",
                    "description": "Random seed for reproducibility"
                }
            },
            "required": ["algorithm", "data", "target", "method", "param_grid"]
        }
    },

    "tuiml_read_data": {
        "name": "tuiml_read_data",
        "description": (
            "Read and preview actual rows from a dataset. Returns sample rows as a list of "
            "dictionaries. Supports head, tail, random sample, or specific row indices."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "data": {
                    "type": "string",
                    "description": "Data file path or built-in dataset name (e.g., 'iris', '/tmp/tuiml_preprocessed/file.csv')"
                },
                "n_rows": {
                    "type": "integer",
                    "default": 10,
                    "description": "Number of rows to return (default: 10, max: 100)"
                },
                "mode": {
                    "type": "string",
                    "enum": ["head", "tail", "sample", "indices"],
                    "default": "head",
                    "description": (
                        "How to select rows:\n"
                        "- head: First n_rows (default)\n"
                        "- tail: Last n_rows\n"
                        "- sample: Random sample of n_rows\n"
                        "- indices: Specific row indices (provide 'indices' parameter)"
                    )
                },
                "indices": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Specific row indices to return (only used when mode='indices')"
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Subset of columns to return (optional, returns all if omitted)"
                },
                "include_target": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include the target column in the output (default: True)"
                },
                "target": {
                    "type": "string",
                    "description": "Target column name (used to label the target in the output)"
                }
            },
            "required": ["data"]
        }
    },
    "tuiml_system_info": {
        "name": "tuiml_system_info",
        "description": (
            "Report details about the TuiML installation on this machine: "
            "installed version, install method (uv tool / pip / editable), "
            "package location, Python executable, platform, and the latest "
            "version available on PyPI. Agents can use this to decide whether "
            "an update is worth running via tuiml_self_update."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "check_latest": {
                    "type": "boolean",
                    "description": "Query PyPI for the latest released version. Defaults to true.",
                    "default": True,
                }
            },
        },
    },
    "tuiml_self_update": {
        "name": "tuiml_self_update",
        "description": (
            "Upgrade the TuiML installation to the latest PyPI release. "
            "Auto-detects the installer (uv tool install vs pip) and runs the "
            "appropriate upgrade command. Returns the command, its stdout / "
            "stderr, and the resulting version. Editable / dev installs are "
            "refused — those need a git pull instead.\n\n"
            "IMPORTANT: the running MCP process is still using the old package "
            "in memory. The client (Claude Desktop, Cursor, etc.) must be "
            "restarted for the new version to take effect."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "dry_run": {
                    "type": "boolean",
                    "description": "Print the command that would be run without executing it.",
                    "default": False,
                },
                "target_version": {
                    "type": "string",
                    "description": "Upgrade to a specific version (e.g. '0.1.3'). Defaults to latest.",
                },
            },
        },
    },
}

DISCOVERY_TOOLS = {
    "tuiml_list": {
        "name": "tuiml_list",
        "description": "List all available TuiML components (algorithms, preprocessors, datasets, features).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["algorithm", "preprocessing", "dataset", "feature", "splitting", "all"],
                    "default": "all",
                    "description": "Category to list"
                },
                "type": {
                    "type": "string",
                    "enum": ["classifier", "regressor", "clusterer", "anomaly", "timeseries"],
                    "description": "Filter algorithms by type. 'anomaly' and 'timeseries' filter by tags within classifiers/regressors."
                },
                "search": {
                    "type": "string",
                    "description": "Search keyword"
                },
                "limit": {
                    "type": "integer",
                    "default": 50,
                    "minimum": 1,
                    "maximum": 200,
                    "description": "Maximum number of results to return (default: 50)"
                },
                "offset": {
                    "type": "integer",
                    "default": 0,
                    "minimum": 0,
                    "description": "Number of results to skip (default: 0)"
                }
            },
            "required": []
        }
    },

    "tuiml_describe": {
        "name": "tuiml_describe",
        "description": "Get detailed information and parameter schema for any TuiML component.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Component name (e.g., 'RandomForestClassifier', 'SimpleImputer', 'iris')"
                }
            },
            "required": ["name"]
        }
    },

    "tuiml_search": {
        "name": "tuiml_search",
        "description": "Search for components by keyword in name or description.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "category": {
                    "type": "string",
                    "enum": ["algorithm", "preprocessing", "dataset", "feature", "all"],
                    "default": "all",
                    "description": "Category to search"
                }
            },
            "required": ["query"]
        }
    }
}

# =============================================================================
# Tool Executors
# =============================================================================

def execute_train(**kwargs) -> Dict[str, Any]:
    """Execute training workflow."""
    import tuiml

    algo_params = kwargs.pop('algorithm_params', {})
    save_path = kwargs.pop('save_path', None)
    kwargs.update(algo_params)

    # Pre-resolve data via the shared loader so dataset_ids, upload paths,
    # and built-in dataset names all work identically (Workflow only handles
    # file paths + built-in names).
    data_arg = kwargs.get('data')
    if isinstance(data_arg, str):
        try:
            kwargs['data'] = _load_data(data_arg)
        except Exception as e:
            return {
                'status': 'error',
                'error': f"Could not resolve data='{data_arg}': {e}",
                'error_type': type(e).__name__,
                'suggestion': (
                    'Use a built-in dataset name (e.g., "iris"), a dataset_id from '
                    'tuiml_upload_data, or an existing file path.'
                )
            }

    try:
        result = tuiml.train(**kwargs)

        # Save model to disk and track by model_id
        model_id = None
        model_path = None
        if result.model:
            model_id = uuid.uuid4().hex[:12]
            model_path = _save_model_to_disk(result.model, model_id, save_path)
            _MODEL_INDEX[model_id] = model_path

        return {
            'status': 'success',
            'model_id': model_id,
            'model_path': model_path,
            'metrics': result.metrics,
            'cv_results': result.cv_results,
            'model_class': result.model.__class__.__name__ if result.model else None,
            'metadata': result.metadata
        }
    except KeyError as e:
        return {
            'status': 'error',
            'error': f"Algorithm not found: {kwargs.get('algorithm')}",
            'error_type': 'KeyError',
            'suggestion': "Use 'tuiml_list' with category='algorithm' to see available algorithms",
            'recovery_tool': 'tuiml_list',
            'recovery_params': {'category': 'algorithm'}
        }
    except ValueError as e:
        return {
            'status': 'error',
            'error': str(e),
            'error_type': 'ValueError',
            'suggestion': "Check parameter types and values. Use 'tuiml_describe' to see the algorithm's parameter schema",
            'recovery_tool': 'tuiml_describe',
            'recovery_params': {'name': kwargs.get('algorithm')}
        }
    except FileNotFoundError as e:
        return {
            'status': 'error',
            'error': f"Data file not found: {kwargs.get('data')}",
            'error_type': 'FileNotFoundError',
            'suggestion': 'Check the file path or use a built-in dataset name (e.g., "iris", "wine")'
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }

def _get_model_tags(model) -> List[str]:
    """Get tags from a model if available."""
    tags = getattr(model, '_tags', [])
    if not tags:
        # Try class-level tags
        tags = getattr(model.__class__, '_tags', [])
    return tags or []


def execute_predict(**kwargs) -> Dict[str, Any]:
    """Execute prediction with support for timeseries and anomaly models."""
    import numpy as np

    try:
        model_id = kwargs.get('model_id')
        model_path = kwargs.get('model_path')

        model = _load_model_from_disk(model_id, model_path)
        if model is None:
            return {
                'status': 'error',
                'error': 'Model not found. Provide model_id (from tuiml_train) or a valid model_path.',
                'error_type': 'ValueError',
                'suggestion': 'Train a model first with tuiml_train which returns a model_id and model_path'
            }

        tags = _get_model_tags(model)

        # Timeseries models: use steps parameter
        if 'timeseries' in tags:
            steps = kwargs.get('steps', 10)
            predictions = model.predict(steps)
            predictions = np.asarray(predictions)
            result = {
                'status': 'success',
                'model_type': 'timeseries',
                'num_predictions': len(predictions),
                'predictions_preview': predictions[:10].tolist(),
                'steps': steps
            }
            if kwargs.get('output_path'):
                np.savetxt(kwargs['output_path'], predictions)
                result['output_path'] = kwargs['output_path']
            return result

        # Anomaly detection models
        if 'anomaly-detection' in tags:
            dataset = _load_data(kwargs['data'])
            predictions = model.predict(dataset.X)
            predictions = np.asarray(predictions)
            result = {
                'status': 'success',
                'model_type': 'anomaly',
                'num_predictions': len(predictions),
                'predictions_preview': predictions[:10].tolist(),
                'n_anomalies': int(np.sum(predictions == -1)),
                'n_normal': int(np.sum(predictions == 1)),
                'anomaly_ratio': float(np.mean(predictions == -1))
            }
            # Get anomaly scores if available
            if hasattr(model, 'decision_function'):
                try:
                    scores = model.decision_function(dataset.X)
                    scores = np.asarray(scores)
                    result['anomaly_scores_preview'] = scores[:10].tolist()
                    result['score_stats'] = {
                        'mean': float(np.mean(scores)),
                        'std': float(np.std(scores)),
                        'min': float(np.min(scores)),
                        'max': float(np.max(scores))
                    }
                except Exception:
                    pass
            if kwargs.get('output_path'):
                np.savetxt(kwargs['output_path'], predictions)
                result['output_path'] = kwargs['output_path']
            return result

        # Standard supervised/clustering prediction
        import tuiml
        dataset = _load_data(kwargs['data'])
        predictions = tuiml.predict(model, dataset.X)

        result = {
            'status': 'success',
            'num_predictions': len(predictions),
            'predictions_preview': predictions[:10].tolist()
        }

        if kwargs.get('output_path'):
            np.savetxt(kwargs['output_path'], predictions)
            result['output_path'] = kwargs['output_path']

        return result
    except FileNotFoundError as e:
        return {
            'status': 'error',
            'error': f"File not found: {str(e)}",
            'error_type': 'FileNotFoundError',
            'suggestion': 'Check the file path or use model_id from tuiml_train instead'
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }

def execute_evaluate(**kwargs) -> Dict[str, Any]:
    """Execute evaluation with support for timeseries and anomaly models."""
    import numpy as np

    try:
        model_id = kwargs.get('model_id')
        model_path = kwargs.get('model_path')

        model = _load_model_from_disk(model_id, model_path)
        if model is None:
            return {
                'status': 'error',
                'error': 'Model not found. Provide model_id (from tuiml_train) or a valid model_path.',
                'error_type': 'ValueError',
                'suggestion': 'Train a model first with tuiml_train which returns a model_id and model_path'
            }

        tags = _get_model_tags(model)
        dataset = _load_data(kwargs['data'])

        # Timeseries evaluation: holdout last 20% as test
        if 'timeseries' in tags:
            from tuiml.evaluation.metrics import (
                mean_absolute_error, mean_squared_error, r2_score
            )

            y = np.asarray(dataset.y) if dataset.y is not None else np.asarray(dataset.X).ravel()
            split_idx = int(len(y) * 0.8)
            y_train, y_test = y[:split_idx], y[split_idx:]

            # Re-fit on training portion
            model.fit(y_train)
            forecast = model.predict(len(y_test))
            forecast = np.asarray(forecast)

            metrics = {
                'mean_absolute_error': float(mean_absolute_error(y_test, forecast)),
                'mean_squared_error': float(mean_squared_error(y_test, forecast)),
                'root_mean_squared_error': float(np.sqrt(mean_squared_error(y_test, forecast))),
            }
            try:
                metrics['r2_score'] = float(r2_score(y_test, forecast))
            except Exception:
                pass

            return {
                'status': 'success',
                'model_type': 'timeseries',
                'metrics': metrics,
                'train_size': int(split_idx),
                'test_size': int(len(y_test)),
                'forecast_preview': forecast[:10].tolist()
            }

        # Anomaly detection evaluation: unsupervised stats + optional supervised
        if 'anomaly-detection' in tags:
            predictions = np.asarray(model.predict(dataset.X))
            n_anomalies = int(np.sum(predictions == -1))
            n_total = len(predictions)

            result = {
                'status': 'success',
                'model_type': 'anomaly',
                'metrics': {
                    'n_anomalies': n_anomalies,
                    'n_normal': int(n_total - n_anomalies),
                    'anomaly_ratio': float(n_anomalies / n_total) if n_total > 0 else 0.0
                }
            }

            # Anomaly scores statistics
            if hasattr(model, 'decision_function'):
                try:
                    scores = np.asarray(model.decision_function(dataset.X))
                    result['metrics']['score_mean'] = float(np.mean(scores))
                    result['metrics']['score_std'] = float(np.std(scores))
                except Exception:
                    pass

            # If ground truth labels available, compute supervised metrics
            if dataset.y is not None:
                try:
                    from tuiml.evaluation.metrics import accuracy_score, precision_score, recall_score, f1_score
                    y_true = np.asarray(dataset.y)
                    result['metrics']['accuracy'] = float(accuracy_score(y_true, predictions))
                    result['metrics']['precision'] = float(precision_score(y_true, predictions))
                    result['metrics']['recall'] = float(recall_score(y_true, predictions))
                    result['metrics']['f1'] = float(f1_score(y_true, predictions))
                except Exception:
                    pass

            return result

        # Standard supervised/clustering evaluation
        import tuiml
        metrics = tuiml.evaluate(
            model, dataset.X, dataset.y,
            metrics=kwargs.get('metrics', 'auto')
        )
        return {'status': 'success', 'metrics': metrics}
    except FileNotFoundError as e:
        return {
            'status': 'error',
            'error': f"File not found: {str(e)}",
            'error_type': 'FileNotFoundError',
            'suggestion': 'Check file paths or use model_id from tuiml_train instead'
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }

def execute_experiment(**kwargs) -> Dict[str, Any]:
    """Execute experiment comparison."""
    import tuiml
    import tuiml.algorithms  # noqa: F401 - trigger registration
    import numpy as np
    from tuiml.evaluation import metrics as metrics_module
    from tuiml.hub import registry, ComponentType

    try:
        progress_callback = kwargs.pop('_progress_callback', None)

        # Collect progress messages
        progress_log = []

        def _on_progress(info):
            progress_log.append(info)
            if progress_callback:
                progress_callback(info)

        # Support single dataset string or list of datasets
        data_input = kwargs['data']
        if isinstance(data_input, str):
            data_names = [data_input]
        else:
            data_names = data_input

        # Load all datasets
        datasets_dict = {}
        for name in data_names:
            ds = _load_data(name)
            datasets_dict[name] = (ds.X, ds.y)

        algorithm_names = kwargs['algorithms']

        # Detect algorithm types to choose appropriate default metrics
        algorithm_types = []
        for algo_name in algorithm_names:
            try:
                if algo_name in registry:
                    info = registry.get_info(algo_name)
                    algorithm_types.append(info.get('type', 'unknown'))
                else:
                    algorithm_types.append('unknown')
            except:
                algorithm_types.append('unknown')

        # Choose default metrics based on algorithm types
        is_clustering = any(t in ['clusterer', 'clustering'] for t in algorithm_types)
        is_regression = any(t in ['regressor', 'regression'] for t in algorithm_types)

        # Validate metric/algorithm compatibility
        if kwargs.get('metrics'):
            user_metrics = kwargs['metrics']
            supervised_metrics = {'accuracy', 'accuracy_score', 'f1', 'f1_score', 'precision',
                                 'precision_score', 'recall', 'recall_score', 'roc_auc'}
            clustering_metrics = {'silhouette', 'silhouette_score', 'calinski_harabasz',
                                 'calinski_harabasz_score', 'davies_bouldin', 'davies_bouldin_score'}

            # Check for mismatch
            has_supervised_metrics = any(m.lower() in supervised_metrics for m in user_metrics)

            if is_clustering and has_supervised_metrics:
                # Warning: supervised metrics requested for clustering
                return {
                    'status': 'error',
                    'error': (
                        'Clustering algorithms require unsupervised metrics. '
                        f'You requested: {user_metrics}, but these are supervised metrics. '
                        'For clustering, use metrics like: '
                        '["silhouette_score", "calinski_harabasz_score", "davies_bouldin_score"]. '
                        'Or omit metrics parameter to use defaults.'
                    ),
                    'suggested_metrics': ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score'],
                    'algorithm_types': algorithm_types
                }

            requested_metrics = user_metrics
        elif is_clustering:
            # Use clustering metrics for clusterers
            requested_metrics = ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score']
        elif is_regression:
            requested_metrics = ['r2_score', 'root_mean_squared_error', 'mean_absolute_error']
        else:
            # Default to classification metrics
            requested_metrics = ['accuracy_score']

        exp = tuiml.experiment(
            algorithms=algorithm_names,
            datasets=datasets_dict,
            cv=kwargs.get('cv', 10),
            metrics=requested_metrics,
            progress_callback=_on_progress
        )

        # Try to get summary, fallback to manual extraction
        try:
            summary = exp.summary() if hasattr(exp, 'summary') else None
        except Exception:
            summary = None

        # Extract results manually for robustness
        results_data = {}
        if hasattr(exp, 'results') and hasattr(exp.results, 'dataset_results'):
            for dataset_name, dataset_result in exp.results.dataset_results.items():
                results_data[dataset_name] = {}
                for model_name, model_result in dataset_result.model_results.items():
                    if model_result.fold_results:
                        # Compute metrics from fold results
                        model_metrics = {}
                        for metric_name in requested_metrics:
                            metric_func = getattr(metrics_module, metric_name, None)
                            if metric_func:
                                scores = []
                                for fold in model_result.fold_results:
                                    try:
                                        score = metric_func(fold.y_true, fold.y_pred)
                                        scores.append(float(score))
                                    except Exception:
                                        pass
                                if scores:
                                    model_metrics[metric_name] = {
                                        'mean': float(np.mean(scores)),
                                        'std': float(np.std(scores)),
                                        'scores': scores
                                    }
                        results_data[dataset_name][model_name] = model_metrics

        result = {
            'status': 'success',
            'summary': summary,
            'results': results_data,
            'algorithms': kwargs['algorithms'],
            'datasets': data_names,
            'cv_folds': kwargs.get('cv', 10)
        }
        if progress_log:
            result['progress_log'] = [
                {
                    'dataset': p.get('dataset'),
                    'model': p.get('model'),
                    'dataset_index': p.get('dataset_index'),
                    'total_datasets': p.get('total_datasets'),
                    'model_index': p.get('model_index'),
                    'total_models': p.get('total_models'),
                    'mean_scores': p.get('mean_scores'),
                }
                for p in progress_log
            ]
        return result
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def execute_list(**kwargs) -> Dict[str, Any]:
    """Execute list components."""
    from tuiml.llm.registry import get_all_tools, list_tools_by_category

    try:
        category = kwargs.get('category', 'all')
        search = kwargs.get('search')
        algo_type = kwargs.get('type')
        limit = kwargs.get('limit', 50)
        offset = kwargs.get('offset', 0)

        if category == 'all':
            tools = get_all_tools()
        else:
            tools = {t.name: t for t in list_tools_by_category(category)}

        # Filter by search
        if search:
            tools = {
                name: tool for name, tool in tools.items()
                if search.lower() in name.lower() or search.lower() in tool.description.lower()
            }

        # Build component list with type/tags from hub registry for algorithms
        from tuiml.hub import registry as hub_registry
        import tuiml.algorithms  # noqa: F401 - trigger registration

        components_list = []
        for t in tools.values():
            entry = {'name': t.name, 'description': t.description, 'category': t.category}

            # For algorithm tools, enrich with type and tags from hub registry
            if t.category == 'algorithm':
                # Strip prefix to get the class name
                class_name = t.name
                for prefix in ('tuiml_algorithm_',):
                    if class_name.startswith(prefix):
                        class_name = class_name[len(prefix):]
                try:
                    info = hub_registry.get_info(class_name)
                    entry['type'] = info.get('type', '')
                    entry['tags'] = info.get('tags', [])
                except (KeyError, Exception):
                    pass

            components_list.append(entry)

        # Filter by algorithm type (classifier, regressor, clusterer, anomaly, timeseries)
        if algo_type:
            if algo_type == 'anomaly':
                components_list = [
                    c for c in components_list
                    if 'anomaly-detection' in c.get('tags', [])
                ]
            elif algo_type == 'timeseries':
                components_list = [
                    c for c in components_list
                    if 'timeseries' in c.get('tags', [])
                ]
            else:
                components_list = [
                    c for c in components_list
                    if c.get('type') == algo_type
                ]

        total = len(components_list)

        # Apply pagination
        paginated = components_list[offset:offset + limit]

        # Format result
        result = {
            'status': 'success',
            'total': total,
            'count': len(paginated),
            'limit': limit,
            'offset': offset,
            'has_more': (offset + limit) < total,
            'components': paginated
        }

        return result
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__,
            'suggestion': 'Check that the category parameter is valid. Use category="all" to list all components.'
        }

def execute_describe(**kwargs) -> Dict[str, Any]:
    """Execute describe component."""
    try:
        name = kwargs['name']

        # 1. Try as algorithm from hub registry (covers all registered algorithms
        #    including community/hub uploads)
        try:
            from tuiml.hub import registry as hub_registry, ComponentType
            import tuiml.algorithms  # noqa: F401 - trigger registration

            component = hub_registry.get(name)
            if component:
                schema = {}
                if hasattr(component, 'get_parameter_schema'):
                    schema = component.get_parameter_schema()

                info = {}
                if hasattr(hub_registry, 'get_info'):
                    try:
                        info = hub_registry.get_info(name)
                    except Exception:
                        pass

                return {
                    'status': 'success',
                    'type': info.get('type', 'algorithm'),
                    'name': name,
                    'description': (component.__doc__ or '').split('\n')[0].strip(),
                    'parameters': schema,
                    'tags': info.get('tags', []),
                    'version': info.get('version', ''),
                }
        except (ImportError, ValueError, KeyError):
            pass

        # 2. Try as dataset
        try:
            from tuiml.datasets.builtin import get_dataset_info
            info = get_dataset_info(name)
            return {
                'status': 'success',
                'type': 'dataset',
                'name': name,
                **info
            }
        except (ValueError, KeyError, ImportError):
            pass

        # 3. Try from component tool registry (preprocessing, features, splitting)
        from tuiml.llm.registry import get_all_tools
        tools = get_all_tools()

        for prefix in ['tuiml_preprocessing_', 'tuiml_feature_', 'tuiml_splitting_']:
            tool = tools.get(f"{prefix}{name}")
            if tool:
                return {
                    'status': 'success',
                    'type': tool.category,
                    'name': name,
                    'description': tool.description,
                    'parameters': tool.input_schema
                }

        return {
            'status': 'error',
            'error': f"Component '{name}' not found",
            'suggestion': "Use 'tuiml_search' to find components by keyword, or 'tuiml_list' to browse all components",
            'recovery_tool': 'tuiml_search',
            'recovery_params': {'query': name}
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }

def execute_search(**kwargs) -> Dict[str, Any]:
    """Execute search components."""
    from tuiml.llm.registry import get_all_tools

    try:
        query = kwargs['query'].lower()
        category = kwargs.get('category', 'all')

        tools = get_all_tools()

        matches = []
        for name, tool in tools.items():
            if category != 'all' and tool.category != category:
                continue

            if query in name.lower() or query in tool.description.lower():
                matches.append({
                    'name': tool.name,
                    'description': tool.description,
                    'category': tool.category
                })

        result = {
            'status': 'success',
            'query': kwargs['query'],
            'count': len(matches),
            'results': matches
        }

        if len(matches) == 0:
            result['suggestion'] = "No matches found. Try a broader search term or use 'tuiml_list' to browse all components"

        return result
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }

def execute_upload_data(**kwargs) -> Dict[str, Any]:
    """Register a dataset file or inline content for use with other tools."""
    import shutil

    try:
        src_path = kwargs.get('file_path')
        content = kwargs.get('content')

        if not src_path and not content:
            return {
                'status': 'error',
                'error': "Provide either 'file_path' (path to CSV/ARFF on disk) or 'content' (inline text).",
                'error_type': 'ValueError'
            }

        upload_dir = _UPLOADS_DIR
        os.makedirs(upload_dir, exist_ok=True)

        if src_path:
            # --- File path mode: validate and copy/link the file ---
            src_path = os.path.expanduser(src_path)
            if not os.path.isfile(src_path):
                return {
                    'status': 'error',
                    'error': f"File not found: {src_path}",
                    'error_type': 'FileNotFoundError'
                }

            ext = os.path.splitext(src_path)[1].lower()
            supported = {'.csv', '.tsv', '.arff', '.parquet', '.pq', '.json', '.jsonl', '.ndjson', '.xlsx', '.xls', '.npy', '.npz'}
            if ext not in supported:
                return {
                    'status': 'error',
                    'error': f"Unsupported file type '{ext}'. Supported: {sorted(supported)}",
                    'error_type': 'ValueError'
                }

            name = kwargs.get('name') or os.path.splitext(os.path.basename(src_path))[0]
            dest_path = os.path.join(upload_dir, f'{name}{ext}')
            shutil.copy2(src_path, dest_path)
            file_path = dest_path
        else:
            # --- Content mode: write inline text to file ---
            file_format = kwargs.get('format', 'csv')
            name = kwargs.get('name', f'uploaded_{uuid.uuid4().hex[:8]}')
            file_path = os.path.join(upload_dir, f'{name}.{file_format}')
            with open(file_path, 'w') as f:
                f.write(content)

        # Validate the file can be loaded
        try:
            dataset = _load_data(file_path)
            n_rows, n_cols = dataset.X.shape if hasattr(dataset, 'X') else (None, None)
            feature_names = list(dataset.feature_names) if hasattr(dataset, 'feature_names') and dataset.feature_names is not None else None
            dataset_id = os.path.splitext(os.path.basename(file_path))[0]
            _DATASET_INDEX[dataset_id] = file_path
            return {
                'status': 'success',
                'dataset_id': dataset_id,
                'file_path': file_path,
                'rows': n_rows,
                'features': n_cols,
                'feature_names': feature_names,
                'message': (
                    f'Dataset registered ({n_rows} rows, {n_cols} features). '
                    f'Pass data="{dataset_id}" (or the full file_path) to other tools.'
                )
            }
        except Exception as e:
            os.remove(file_path)
            return {
                'status': 'error',
                'error': f'Invalid dataset: {str(e)}',
                'error_type': type(e).__name__,
                'suggestion': 'Ensure the file is a valid CSV (with header row) or ARFF file.'
            }

    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }

def execute_save_model(**kwargs) -> Dict[str, Any]:
    """Copy a trained model to a user-specified location."""
    import shutil

    try:
        model_id = kwargs['model_id']
        destination = kwargs['destination']

        if model_id not in _MODEL_INDEX:
            return {
                'status': 'error',
                'error': f"Model '{model_id}' not found",
                'error_type': 'KeyError',
                'suggestion': 'Train a model first with tuiml_train which returns a model_id'
            }

        source = _MODEL_INDEX[model_id]
        os.makedirs(os.path.dirname(os.path.abspath(destination)) or '.', exist_ok=True)
        shutil.copy2(source, destination)

        return {
            'status': 'success',
            'model_id': model_id,
            'source': source,
            'destination': os.path.abspath(destination),
            'message': f'Model saved to {os.path.abspath(destination)}'
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }

def execute_serve_model(**kwargs) -> Dict[str, Any]:
    """Start a REST API server to serve a trained model."""
    import threading
    import socket

    try:
        model_id = kwargs.get('model_id')
        model_path = kwargs.get('model_path')
        port = kwargs.get('port', 8000)
        host = kwargs.get('host', '127.0.0.1')

        # Resolve model file path
        if model_id and model_id in _MODEL_INDEX:
            serve_path = _MODEL_INDEX[model_id]
        elif model_path and os.path.exists(model_path):
            serve_path = model_path
            model_id = model_id or os.path.splitext(os.path.basename(model_path))[0]
        else:
            return {
                'status': 'error',
                'error': 'Model not found. Provide model_id (from tuiml_train) or a valid model_path.',
                'error_type': 'ValueError',
                'suggestion': 'Train a model first with tuiml_train'
            }

        # Check port availability
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex((host, port)) == 0:
                return {
                    'status': 'error',
                    'error': f'Port {port} is already in use',
                    'error_type': 'OSError',
                    'suggestion': f'Use a different port or stop the existing server with tuiml_stop_server'
                }

        # Check dependencies
        try:
            from tuiml.serving.server import ModelServer
            import uvicorn
        except ImportError:
            return {
                'status': 'error',
                'error': 'Serving dependencies not installed',
                'error_type': 'ImportError',
                'suggestion': 'Install with: pip install "tuiml[serving]" (requires fastapi and uvicorn)'
            }

        # Create server and load model
        model_server = ModelServer()
        model_server.load_model(model_id, serve_path)
        app = model_server.create_app()

        # Configure uvicorn to run in background thread
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="warning",
        )
        server = uvicorn.Server(config)

        server_id = uuid.uuid4().hex[:8]
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        # Wait briefly for server to start
        import time
        time.sleep(1.0)

        base_url = f"http://{host}:{port}"

        _SERVING_SERVERS[server_id] = {
            'server': server,
            'thread': thread,
            'model_server': model_server,
            'model_id': model_id,
            'model_path': serve_path,
            'host': host,
            'port': port,
            'url': base_url,
        }

        return {
            'status': 'success',
            'server_id': server_id,
            'model_id': model_id,
            'url': base_url,
            'endpoints': {
                'predict': f'{base_url}/predict',
                'predict_model': f'{base_url}/models/{model_id}/predict',
                'predict_proba': f'{base_url}/models/{model_id}/predict_proba',
                'health': f'{base_url}/health',
                'models': f'{base_url}/models',
                'docs': f'{base_url}/docs',
            },
            'example_curl': (
                f'curl -X POST {base_url}/predict '
                f'-H "Content-Type: application/json" '
                f'-d \'{{"features": [[5.1, 3.5, 1.4, 0.2]]}}\''
            ),
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }


def execute_stop_server(**kwargs) -> Dict[str, Any]:
    """Stop a running model serving server."""
    try:
        server_id = kwargs.get('server_id')

        if server_id:
            # Stop specific server
            if server_id not in _SERVING_SERVERS:
                return {
                    'status': 'error',
                    'error': f"Server '{server_id}' not found",
                    'suggestion': 'Use tuiml_server_status to see running servers'
                }
            info = _SERVING_SERVERS.pop(server_id)
            info['server'].should_exit = True
            info['thread'].join(timeout=5)
            return {
                'status': 'success',
                'message': f"Server {server_id} stopped (was serving {info['model_id']} on port {info['port']})"
            }
        else:
            # Stop all servers
            stopped = []
            for sid, info in list(_SERVING_SERVERS.items()):
                info['server'].should_exit = True
                info['thread'].join(timeout=5)
                stopped.append({'server_id': sid, 'model_id': info['model_id'], 'port': info['port']})
            _SERVING_SERVERS.clear()
            return {
                'status': 'success',
                'message': f'Stopped {len(stopped)} server(s)',
                'stopped': stopped
            }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }


def execute_server_status(**kwargs) -> Dict[str, Any]:
    """Get status of running model serving servers."""
    servers = []
    for sid, info in _SERVING_SERVERS.items():
        servers.append({
            'server_id': sid,
            'model_id': info['model_id'],
            'model_path': info['model_path'],
            'url': info['url'],
            'port': info['port'],
            'running': info['thread'].is_alive(),
        })
    return {
        'status': 'success',
        'count': len(servers),
        'servers': servers
    }


def execute_plot(**kwargs) -> Dict[str, Any]:
    """Execute a visualization/plot generation."""
    import base64
    import numpy as np

    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        plot_type = kwargs['plot_type']
        title = kwargs.get('title')

        # Create a temp file for the plot
        plot_path = os.path.join(tempfile.gettempdir(), f'tuiml_plot_{uuid.uuid4().hex[:8]}.png')

        if plot_type == 'confusion_matrix':
            from tuiml.evaluation.visualization import plot_confusion_matrix
            import tuiml

            model = _load_model_from_disk(kwargs.get('model_id'), kwargs.get('model_path'))
            if model is None:
                return {'status': 'error', 'error': 'Model not found. Provide model_id or model_path.'}

            dataset = _load_data(kwargs['data'])
            predictions = tuiml.predict(model, dataset.X)
            plot_confusion_matrix(
                dataset.y, predictions,
                title=title or 'Confusion Matrix',
                save_path=plot_path,
                normalize=kwargs.get('normalize', False),
            )
            description = 'Confusion matrix showing predicted vs actual class labels.'

        elif plot_type == 'roc_curve':
            from tuiml.evaluation.visualization import plot_roc_curve
            import tuiml

            model = _load_model_from_disk(kwargs.get('model_id'), kwargs.get('model_path'))
            if model is None:
                return {'status': 'error', 'error': 'Model not found. Provide model_id or model_path.'}

            dataset = _load_data(kwargs['data'])

            if not hasattr(model, 'predict_proba'):
                return {
                    'status': 'error',
                    'error': 'Model does not support predict_proba, required for ROC curve.',
                    'suggestion': 'Use a classifier that supports probability estimates (e.g., RandomForestClassifier, NaiveBayesClassifier).'
                }
            probas = model.predict_proba(dataset.X)
            # For binary classification, use the probability of the positive class
            if probas.ndim == 2:
                y_score = probas[:, 1]
            else:
                y_score = probas

            plot_roc_curve(
                dataset.y, y_score,
                title=title or 'ROC Curve',
                save_path=plot_path,
            )
            description = 'ROC curve with AUC score.'

        elif plot_type == 'pr_curve':
            from tuiml.evaluation.visualization import plot_pr_curve
            import tuiml

            model = _load_model_from_disk(kwargs.get('model_id'), kwargs.get('model_path'))
            if model is None:
                return {'status': 'error', 'error': 'Model not found. Provide model_id or model_path.'}

            dataset = _load_data(kwargs['data'])

            if not hasattr(model, 'predict_proba'):
                return {
                    'status': 'error',
                    'error': 'Model does not support predict_proba, required for PR curve.',
                    'suggestion': 'Use a classifier that supports probability estimates.'
                }
            probas = model.predict_proba(dataset.X)
            if probas.ndim == 2:
                y_score = probas[:, 1]
            else:
                y_score = probas

            plot_pr_curve(
                dataset.y, y_score,
                title=title or 'Precision-Recall Curve',
                save_path=plot_path,
            )
            description = 'Precision-Recall curve with Average Precision score.'

        elif plot_type == 'learning_curve':
            from tuiml.evaluation.visualization import plot_learning_curve
            from tuiml.hub import registry
            from tuiml.evaluation.metrics import accuracy_score
            import tuiml.algorithms  # noqa: F401 - trigger registration

            algorithm_name = kwargs.get('algorithm')
            if not algorithm_name:
                return {'status': 'error', 'error': 'algorithm parameter is required for learning_curve.'}

            dataset = _load_data(kwargs['data'])

            algo_cls = registry.get(algorithm_name)
            if algo_cls is None:
                return {'status': 'error', 'error': f"Algorithm '{algorithm_name}' not found."}

            # Compute learning curve manually with k-fold CV
            from tuiml.evaluation.splitting import KFold
            n_splits = 5
            train_fractions = np.linspace(0.1, 1.0, 10)
            n_samples = len(dataset.y)
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

            all_train_sizes = []
            all_train_scores = []  # shape: (n_sizes, n_splits)
            all_test_scores = []

            for frac in train_fractions:
                fold_train_scores = []
                fold_test_scores = []
                for train_idx, test_idx in kf.split(dataset.X):
                    X_train_full, y_train_full = dataset.X[train_idx], dataset.y[train_idx]
                    X_test, y_test = dataset.X[test_idx], dataset.y[test_idx]

                    # Subsample training set
                    subset_size = max(2, int(len(X_train_full) * frac))
                    X_train = X_train_full[:subset_size]
                    y_train = y_train_full[:subset_size]

                    model_lc = algo_cls()
                    model_lc.fit(X_train, y_train)

                    train_pred = model_lc.predict(X_train)
                    test_pred = model_lc.predict(X_test)
                    fold_train_scores.append(accuracy_score(y_train, train_pred))
                    fold_test_scores.append(accuracy_score(y_test, test_pred))

                # Use the actual training size for the first fold as representative
                all_train_sizes.append(max(2, int(len(dataset.X) * (n_splits - 1) / n_splits * frac)))
                all_train_scores.append(fold_train_scores)
                all_test_scores.append(fold_test_scores)

            train_sizes_arr = np.array(all_train_sizes)
            train_scores_arr = np.array(all_train_scores)
            test_scores_arr = np.array(all_test_scores)

            plot_learning_curve(
                train_sizes_arr,
                train_scores_arr,
                test_scores_arr,
                title=title or 'Learning Curve',
                save_path=plot_path,
                metric_name='Accuracy',
            )
            description = f'Learning curve for {algorithm_name} showing training vs validation accuracy.'

        elif plot_type == 'tree':
            from tuiml.evaluation.visualization import plot_tree

            model = _load_model_from_disk(kwargs.get('model_id'), kwargs.get('model_path'))
            if model is None:
                return {'status': 'error', 'error': 'Model not found. Provide model_id or model_path.'}

            # Get feature names from the model if available
            feature_names = None
            if hasattr(model, 'feature_names_'):
                feature_names = model.feature_names_

            plot_tree(
                model,
                feature_names=feature_names,
                filled=True,
                rounded=True,
                title=title or 'Decision Tree',
                save_path=plot_path,
            )
            description = 'Decision tree structure visualization.'

        elif plot_type == 'feature_importance':
            model = _load_model_from_disk(kwargs.get('model_id'), kwargs.get('model_path'))
            if model is None:
                return {'status': 'error', 'error': 'Model not found. Provide model_id or model_path.'}

            importances = None

            # Try direct attribute first
            if hasattr(model, 'feature_importances_') and model.feature_importances_ is not None:
                importances = np.array(model.feature_importances_)

            # Try wrapped inner model (e.g., XGBoost, GradientBoosting store
            # the sklearn-compatible model in model.model_)
            if importances is None and hasattr(model, 'model_'):
                inner = model.model_
                if hasattr(inner, 'feature_importances_') and inner.feature_importances_ is not None:
                    importances = np.array(inner.feature_importances_)

            # Try coef_ for linear models (Logistic Regression, SVM, etc.)
            if importances is None:
                coef = getattr(model, 'coef_', None)
                if coef is None and hasattr(model, 'model_'):
                    coef = getattr(model.model_, 'coef_', None)
                if coef is not None:
                    coef = np.array(coef)
                    if coef.ndim > 1:
                        importances = np.mean(np.abs(coef), axis=0)
                    else:
                        importances = np.abs(coef)

            def _count_feature_usage(node, counts):
                """Recursively count feature usage in a tree node."""
                if node is None or getattr(node, 'is_leaf', True):
                    return
                feat = getattr(node, 'feature_index', None)
                if feat is not None and feat >= 0:
                    samples = getattr(node, 'n_samples', 1)
                    counts[feat] += samples
                _count_feature_usage(getattr(node, 'left', None), counts)
                _count_feature_usage(getattr(node, 'right', None), counts)

            # For ensemble models, compute from estimators' trees
            if importances is None and hasattr(model, 'estimators_') and model.estimators_:
                n_features = getattr(model, 'n_features_', None)
                if n_features and all(hasattr(est, 'tree_') for est in model.estimators_):
                    total = np.zeros(n_features)
                    for est in model.estimators_:
                        _count_feature_usage(est.tree_, total)
                    if total.sum() > 0:
                        importances = total / total.sum()

            # For single tree models
            if importances is None and hasattr(model, 'tree_'):
                n_features = getattr(model, 'n_features_', None)
                if n_features:
                    total = np.zeros(n_features)
                    _count_feature_usage(model.tree_, total)
                    if total.sum() > 0:
                        importances = total / total.sum()

            if importances is None:
                return {
                    'status': 'error',
                    'error': 'Cannot compute feature importances from this model.',
                    'suggestion': 'Use a tree-based model (e.g., RandomForestClassifier, XGBoostClassifier, C45TreeClassifier).'
                }

            importances = np.array(importances)
            feature_names = None
            if hasattr(model, 'feature_names_'):
                feature_names = model.feature_names_
            if feature_names is None and hasattr(model, 'model_'):
                inner = model.model_
                if hasattr(inner, 'feature_names_in_'):
                    feature_names = list(inner.feature_names_in_)
            if feature_names is None:
                feature_names = [f'Feature {i}' for i in range(len(importances))]

            # Sort by importance
            indices = np.argsort(importances)[::-1]
            sorted_names = [feature_names[i] for i in indices]
            sorted_importances = importances[indices]

            from tuiml.evaluation.visualization import setup_figure, style_axis, get_colors
            fig, ax = setup_figure(figsize=(10, max(6, len(sorted_names) * 0.35)))
            colors = get_colors(len(sorted_names))

            ax.barh(range(len(sorted_names)), sorted_importances[::-1], color=colors[0])
            ax.set_yticks(range(len(sorted_names)))
            ax.set_yticklabels(sorted_names[::-1])
            style_axis(
                ax,
                title=title or 'Feature Importance',
                xlabel='Importance',
                ylabel=None,
                legend=False,
            )
            fig.tight_layout()
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            description = 'Feature importance bar chart from model.'

        elif plot_type in ('cd_diagram', 'boxplot_comparison', 'heatmap', 'ranking_table'):
            experiment_results = kwargs.get('experiment_results')
            if not experiment_results:
                return {
                    'status': 'error',
                    'error': f"'{plot_type}' requires experiment_results parameter with algorithm CV scores.",
                    'suggestion': "Provide experiment_results: { 'AlgoName': [score1, score2, ...], ... }"
                }

            scores_dict = {
                name: np.array(scores) for name, scores in experiment_results.items()
            }

            if plot_type == 'cd_diagram':
                from tuiml.evaluation.visualization import plot_critical_difference
                plot_critical_difference(
                    scores=scores_dict,
                    title=title or 'Critical Difference Diagram',
                    save_path=plot_path,
                )
                description = 'Critical difference diagram showing statistically significant differences between algorithms.'

            elif plot_type == 'boxplot_comparison':
                from tuiml.evaluation.visualization import plot_boxplot_comparison
                plot_boxplot_comparison(
                    scores=scores_dict,
                    save_path=plot_path,
                )
                description = 'Box plot comparison of algorithm cross-validation scores.'

            elif plot_type == 'heatmap':
                from tuiml.evaluation.visualization import plot_heatmap
                plot_heatmap(
                    scores=scores_dict,
                    save_path=plot_path,
                )
                description = 'Heatmap of algorithm scores across datasets.'

            elif plot_type == 'ranking_table':
                from tuiml.evaluation.visualization import plot_ranking_table
                plot_ranking_table(
                    scores=scores_dict,
                    title=title or 'Algorithm Ranking',
                    save_path=plot_path,
                )
                description = 'Ranking table of algorithm performance.'

        else:
            return {'status': 'error', 'error': f"Unknown plot_type: '{plot_type}'"}

        # Close any remaining figures to free memory
        plt.close('all')

        # Read the saved plot and base64 encode it
        with open(plot_path, 'rb') as f:
            image_bytes = f.read()
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')

        # Clean up temp file
        os.remove(plot_path)

        return {
            'status': 'success',
            'plot_type': plot_type,
            'description': description,
            '_image_base64': image_b64,
            '_image_mime': 'image/png',
        }

    except Exception as e:
        plt.close('all')
        return {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }


def execute_data_profile(**kwargs) -> Dict[str, Any]:
    """Profile a dataset: shape, dtypes, missing values, stats, class distribution."""
    import numpy as np

    try:
        dataset = _load_data(kwargs['data'])
        X = np.asarray(dataset.X)
        y = dataset.y
        feature_names = list(dataset.feature_names) if hasattr(dataset, 'feature_names') and dataset.feature_names is not None else [f'feature_{i}' for i in range(X.shape[1])]

        result = {
            'status': 'success',
            'name': kwargs['data'],
            'shape': list(X.shape),
            'n_samples': int(X.shape[0]),
            'n_features': int(X.shape[1]),
            'feature_names': feature_names,
        }

        # Dtypes
        dtypes = {}
        for i, name in enumerate(feature_names):
            col = X[:, i]
            try:
                col.astype(float)
                dtypes[name] = 'numeric'
            except (ValueError, TypeError):
                dtypes[name] = 'categorical'
        result['dtypes'] = dtypes

        # Missing values
        missing = {}
        for i, name in enumerate(feature_names):
            col = X[:, i]
            n_missing = int(np.sum(np.isnan(col))) if np.issubdtype(col.dtype, np.number) else 0
            if n_missing > 0:
                missing[name] = n_missing
        result['missing_values'] = missing

        # Numeric stats
        numeric_stats = {}
        for i, name in enumerate(feature_names):
            if dtypes.get(name) == 'numeric':
                col = X[:, i].astype(float)
                valid = col[~np.isnan(col)]
                if len(valid) > 0:
                    numeric_stats[name] = {
                        'mean': float(np.mean(valid)),
                        'std': float(np.std(valid)),
                        'min': float(np.min(valid)),
                        'max': float(np.max(valid)),
                        'median': float(np.median(valid)),
                    }
        result['numeric_stats'] = numeric_stats

        # Class distribution (if target provided)
        target_col = kwargs.get('target')
        if y is not None:
            y_arr = np.asarray(y)
            unique, counts = np.unique(y_arr, return_counts=True)
            result['class_distribution'] = {str(u): int(c) for u, c in zip(unique, counts)}
            if target_col:
                result['target_column'] = target_col

        return result
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }


def execute_generate_data(**kwargs) -> Dict[str, Any]:
    """Generate synthetic data using a generator class."""
    import numpy as np

    try:
        generator_name = kwargs['generator']

        from tuiml.datasets.generators import (
            RandomRBF, Agrawal, LED, Hyperplane,
            Friedman, MexicanHat, Sine,
            Blobs, Moons, Circles, SwissRoll,
        )

        generators = {
            'RandomRBF': RandomRBF, 'Agrawal': Agrawal, 'LED': LED, 'Hyperplane': Hyperplane,
            'Friedman': Friedman, 'MexicanHat': MexicanHat, 'Sine': Sine,
            'Blobs': Blobs, 'Moons': Moons, 'Circles': Circles, 'SwissRoll': SwissRoll,
        }

        gen_cls = generators.get(generator_name)
        if gen_cls is None:
            return {
                'status': 'error',
                'error': f"Generator '{generator_name}' not found.",
                'suggestion': f"Available generators: {list(generators.keys())}"
            }

        # Build constructor params
        params = {}
        extra_params = kwargs.get('generator_params', {})
        for key in ('n_samples', 'n_features', 'n_classes', 'n_clusters', 'noise', 'random_state'):
            if key in kwargs and kwargs[key] is not None:
                params[key] = kwargs[key]
        params.update(extra_params)

        gen = gen_cls(**params)
        data = gen.generate()

        # Save to CSV temp file
        upload_dir = os.path.join(tempfile.gettempdir(), 'tuiml_generated')
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, f'{generator_name.lower()}_{uuid.uuid4().hex[:8]}.csv')

        import pandas as pd
        feature_names = list(data.feature_names) if data.feature_names else [f'x{i}' for i in range(data.X.shape[1])]
        df = pd.DataFrame(data.X, columns=feature_names)
        if data.y is not None:
            df['target'] = data.y
        df.to_csv(file_path, index=False)

        # Preview: first 5 rows
        preview = {col: df[col].head(5).tolist() for col in df.columns[:6]}

        result = {
            'status': 'success',
            'generator': generator_name,
            'file_path': file_path,
            'shape': [int(data.X.shape[0]), int(data.X.shape[1])],
            'feature_names': feature_names,
            'preview': preview,
        }
        if data.target_names:
            result['target_names'] = list(data.target_names)

        return result
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }


def execute_preprocess(**kwargs) -> Dict[str, Any]:
    """Apply preprocessing steps to a dataset."""
    import numpy as np

    try:
        dataset = _load_data(kwargs['data'])
        X = np.asarray(dataset.X, dtype=float)
        y = dataset.y
        original_shape = list(X.shape)
        feature_names = list(dataset.feature_names) if hasattr(dataset, 'feature_names') and dataset.feature_names is not None else [f'feature_{i}' for i in range(X.shape[1])]

        steps = kwargs['steps']
        steps_applied = []

        from tuiml.hub import registry
        import tuiml.preprocessing  # noqa: F401 - trigger registration

        for step in steps:
            if isinstance(step, str):
                name, params = step, {}
            elif isinstance(step, dict):
                name = step.get('name')
                params = {k: v for k, v in step.items() if k != 'name'}
            else:
                continue

            # Resolve preprocessor class
            preprocessor_cls = None
            try:
                preprocessor_cls = registry.get(name)
            except (KeyError, Exception):
                pass

            if preprocessor_cls is None:
                # Fallback: try direct import
                try:
                    from tuiml import preprocessing as pp_module
                    preprocessor_cls = getattr(pp_module, name, None)
                except ImportError:
                    pass

            if preprocessor_cls is None:
                return {
                    'status': 'error',
                    'error': f"Preprocessor '{name}' not found.",
                    'suggestion': "Use tuiml_list with category='preprocessing' to see available preprocessors."
                }

            preprocessor = preprocessor_cls(**params)
            if hasattr(preprocessor, 'fit_resample') and y is not None:
                X, y = preprocessor.fit_resample(X, y)
            else:
                from tuiml.base.preprocessing import InstanceTransformer
                if isinstance(preprocessor, InstanceTransformer):
                    result = preprocessor.fit_transform(X, y)
                    X, y = result[0], result[1]
                else:
                    X = preprocessor.fit_transform(X)

            steps_applied.append(name)

        # Save result to CSV
        import pandas as pd
        save_as = kwargs.get('save_as')
        if save_as:
            file_path = save_as
            os.makedirs(os.path.dirname(os.path.abspath(save_as)) or '.', exist_ok=True)
        else:
            upload_dir = os.path.join(tempfile.gettempdir(), 'tuiml_preprocessed')
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, f'preprocessed_{uuid.uuid4().hex[:8]}.csv')

        # Build output DataFrame
        out_cols = feature_names[:X.shape[1]] if len(feature_names) >= X.shape[1] else [f'feature_{i}' for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=out_cols)
        if y is not None:
            target_name = kwargs.get('target', 'target')
            df[target_name] = y
        df.to_csv(file_path, index=False)

        return {
            'status': 'success',
            'file_path': file_path,
            'original_shape': original_shape,
            'new_shape': list(X.shape),
            'steps_applied': steps_applied,
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }


def execute_select_features(**kwargs) -> Dict[str, Any]:
    """Run feature selection on a dataset."""
    import numpy as np

    try:
        dataset = _load_data(kwargs['data'])
        X = np.asarray(dataset.X, dtype=float)
        y = np.asarray(dataset.y) if dataset.y is not None else None
        feature_names = list(dataset.feature_names) if hasattr(dataset, 'feature_names') and dataset.feature_names is not None else [f'feature_{i}' for i in range(X.shape[1])]

        method_name = kwargs['method']

        from tuiml.features.selection import (
            SelectKBestSelector, SelectPercentileSelector,
            VarianceThresholdSelector, CFSSelector,
            WrapperSelector, SelectFprSelector, SelectThresholdSelector
        )

        selectors = {
            'SelectKBestSelector': SelectKBestSelector,
            'SelectPercentileSelector': SelectPercentileSelector,
            'VarianceThresholdSelector': VarianceThresholdSelector,
            'CFSSelector': CFSSelector,
            'WrapperSelector': WrapperSelector,
            'SelectFprSelector': SelectFprSelector,
            'SelectThresholdSelector': SelectThresholdSelector,
        }

        selector_cls = selectors.get(method_name)
        if selector_cls is None:
            return {
                'status': 'error',
                'error': f"Feature selection method '{method_name}' not found.",
                'suggestion': f"Available methods: {list(selectors.keys())}"
            }

        # Build params
        params = kwargs.get('method_params', {})
        if 'k' in kwargs and kwargs['k'] is not None:
            params['k'] = kwargs['k']
        if 'threshold' in kwargs and kwargs['threshold'] is not None:
            params['threshold'] = kwargs['threshold']

        selector = selector_cls(**params)
        X_selected = selector.fit_transform(X, y)

        # Get selected feature names
        try:
            selected_indices = selector.get_support(indices=True)
            selected_names = [feature_names[i] for i in selected_indices]
        except Exception:
            selected_names = [f'feature_{i}' for i in range(X_selected.shape[1])]

        result = {
            'status': 'success',
            'method': method_name,
            'n_original': int(X.shape[1]),
            'n_selected': int(X_selected.shape[1]),
            'selected_features': selected_names,
        }

        # Include scores if available
        if hasattr(selector, 'scores_') and selector.scores_ is not None:
            scores_arr = np.asarray(selector.scores_)
            result['scores'] = {
                feature_names[i]: float(scores_arr[i])
                for i in range(min(len(scores_arr), len(feature_names)))
            }

        # Save filtered dataset to temp CSV
        import pandas as pd
        upload_dir = os.path.join(tempfile.gettempdir(), 'tuiml_selected')
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, f'selected_{uuid.uuid4().hex[:8]}.csv')

        df = pd.DataFrame(X_selected, columns=selected_names)
        if y is not None:
            target_name = kwargs.get('target', 'target')
            df[target_name] = y
        df.to_csv(file_path, index=False)
        result['file_path'] = file_path

        return result
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }


def execute_statistical_test(**kwargs) -> Dict[str, Any]:
    """Run statistical significance tests on experiment results."""
    import numpy as np

    try:
        test_name = kwargs['test']
        raw_results = kwargs['results']
        alpha = kwargs.get('significance_level', 0.05)
        higher_better = kwargs.get('higher_better', True)

        # Convert results to numpy arrays
        results = {name: np.array(scores, dtype=float) for name, scores in raw_results.items()}

        from tuiml.evaluation.statistics import (
            friedman_test, nemenyi_post_hoc, wilcoxon_signed_rank_test,
            paired_t_test, one_way_anova, friedman_aligned_ranks_test, quade_test,
        )

        if test_name == 'friedman':
            statistic, p_value, significant = friedman_test(results, significance_level=alpha)
            return {
                'status': 'success',
                'test': 'friedman',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': bool(significant),
            }

        elif test_name == 'nemenyi':
            pairwise = nemenyi_post_hoc(results, significance_level=alpha)
            details = {f"{k[0]} vs {k[1]}": bool(v) for k, v in pairwise.items()}
            return {
                'status': 'success',
                'test': 'nemenyi',
                'significant': any(pairwise.values()),
                'details': details,
            }

        elif test_name in ('wilcoxon', 'paired_t'):
            # Pairwise tests: use first two algorithms
            names = list(results.keys())
            if len(names) < 2:
                return {
                    'status': 'error',
                    'error': 'Pairwise tests require at least 2 algorithms in results.',
                }
            x, y = results[names[0]], results[names[1]]

            if test_name == 'wilcoxon':
                stats = wilcoxon_signed_rank_test(x, y, significance_level=alpha, higher_better=higher_better)
            else:
                stats = paired_t_test(x, y, significance_level=alpha, higher_better=higher_better)

            return {
                'status': 'success',
                'test': test_name,
                'algorithms': [names[0], names[1]],
                'statistic': float(stats.t_statistic),
                'p_value': float(stats.p_value),
                'significant': stats.is_significant(),
                'details': {
                    f'{names[0]}_mean': float(stats.x_mean),
                    f'{names[1]}_mean': float(stats.y_mean),
                    f'{names[0]}_std': float(stats.x_std),
                    f'{names[1]}_std': float(stats.y_std),
                    'diff_mean': float(stats.diff_mean),
                    'significance': stats.significance.name,
                }
            }

        elif test_name == 'anova':
            groups = list(results.values())
            f_stat, p_value, significant = one_way_anova(*groups, significance_level=alpha)
            return {
                'status': 'success',
                'test': 'anova',
                'statistic': float(f_stat),
                'p_value': float(p_value),
                'significant': bool(significant),
            }

        elif test_name == 'friedman_aligned':
            statistic, p_value, significant = friedman_aligned_ranks_test(results, significance_level=alpha)
            return {
                'status': 'success',
                'test': 'friedman_aligned',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': bool(significant),
            }

        elif test_name == 'quade':
            statistic, p_value, significant = quade_test(results, significance_level=alpha)
            return {
                'status': 'success',
                'test': 'quade',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': bool(significant),
            }

        else:
            return {
                'status': 'error',
                'error': f"Unknown test: '{test_name}'",
                'suggestion': "Available tests: friedman, nemenyi, wilcoxon, paired_t, anova, friedman_aligned, quade"
            }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }


def execute_tune(**kwargs) -> Dict[str, Any]:
    """Hyperparameter optimization for any algorithm."""
    import numpy as np

    try:
        algorithm_name = kwargs['algorithm']
        progress_callback = kwargs.pop('_progress_callback', None)

        from tuiml.hub import registry
        import tuiml.algorithms  # noqa: F401 - trigger registration

        algo_cls = registry.get(algorithm_name)
        if algo_cls is None:
            return {
                'status': 'error',
                'error': f"Algorithm '{algorithm_name}' not found.",
                'suggestion': "Use tuiml_list with category='algorithm' to see available algorithms."
            }

        dataset = _load_data(kwargs['data'])
        X = np.asarray(dataset.X, dtype=float)
        y = np.asarray(dataset.y)

        method = kwargs['method']
        param_grid = kwargs['param_grid']
        cv = kwargs.get('cv', 5)
        scoring = kwargs.get('scoring', 'accuracy')
        random_state = kwargs.get('random_state')

        # Collect progress messages
        progress_log = []

        def _on_progress(info):
            progress_log.append(info)
            if progress_callback:
                progress_callback(info)

        estimator = algo_cls()

        if method == 'grid':
            from tuiml.evaluation.tuning import GridSearchCV
            tuner = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,
                random_state=random_state,
                progress_callback=_on_progress,
            )
        elif method == 'random':
            from tuiml.evaluation.tuning import RandomSearchCV
            n_iter = kwargs.get('n_iter', 10)
            tuner = RandomSearchCV(
                estimator=estimator,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                random_state=random_state,
                progress_callback=_on_progress,
            )
        elif method == 'bayesian':
            from tuiml.evaluation.tuning import BayesianSearchCV
            n_iterations = kwargs.get('n_iterations', 50)
            tuner = BayesianSearchCV(
                estimator=estimator,
                param_space=param_grid,
                n_iterations=n_iterations,
                cv=cv,
                scoring=scoring,
                random_state=random_state,
                progress_callback=_on_progress,
            )
        else:
            return {
                'status': 'error',
                'error': f"Unknown tuning method: '{method}'",
                'suggestion': "Available methods: 'grid', 'random', 'bayesian'"
            }

        tuner.fit(X, y)

        # Save best estimator
        model_id = uuid.uuid4().hex[:12]
        model_path = _save_model_to_disk(tuner.best_estimator_, model_id)
        _MODEL_INDEX[model_id] = model_path

        # Summarize cv_results
        cv_results_summary = {}
        if hasattr(tuner, 'cv_results_') and tuner.cv_results_:
            cv_res = tuner.cv_results_
            if 'params' in cv_res and 'mean_test_score' in cv_res:
                cv_results_summary['n_candidates'] = len(cv_res['params'])
                cv_results_summary['best_rank'] = int(cv_res.get('rank_test_score', [1])[0]) if 'rank_test_score' in cv_res else 1
                # Top 5 parameter sets
                top_indices = np.argsort(cv_res['mean_test_score'])[::-1][:5]
                cv_results_summary['top_5'] = [
                    {
                        'params': cv_res['params'][i],
                        'mean_score': float(cv_res['mean_test_score'][i]),
                        'std_score': float(cv_res['std_test_score'][i]) if 'std_test_score' in cv_res else 0.0,
                    }
                    for i in top_indices
                ]

        result = {
            'status': 'success',
            'method': method,
            'best_params': tuner.best_params_,
            'best_score': float(tuner.best_score_),
            'cv_results': cv_results_summary,
            'model_id': model_id,
            'model_path': model_path,
        }
        if progress_log:
            result['progress_log'] = [
                {
                    'iteration': p.get('iteration'),
                    'total': p.get('total'),
                    'mean_score': round(p.get('mean_score', 0), 4),
                    'best_score': round(p.get('best_score', 0), 4),
                    'params': p.get('params'),
                }
                for p in progress_log
            ]
        return result
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }


def execute_read_data(**kwargs) -> Dict[str, Any]:
    """Read and preview actual rows from a dataset using Dataset.to_pandas()."""
    try:
        dataset = _load_data(kwargs['data'])
        include_target = kwargs.get('include_target', True)
        df = dataset.to_pandas(include_target=include_target)

        n_rows = min(kwargs.get('n_rows', 10), 100)
        mode = kwargs.get('mode', 'head')

        # Filter columns if requested
        requested_cols = kwargs.get('columns')
        if requested_cols:
            # Keep only columns that exist
            valid = [c for c in requested_cols if c in df.columns]
            # Always include target if present and requested
            target_name = kwargs.get('target')
            if include_target and target_name and target_name in df.columns and target_name not in valid:
                valid.append(target_name)
            df = df[valid]

        # Select rows based on mode
        if mode == 'head':
            subset = df.head(n_rows)
        elif mode == 'tail':
            subset = df.tail(n_rows)
        elif mode == 'sample':
            subset = df.sample(n=min(n_rows, len(df)), random_state=42)
        elif mode == 'indices':
            indices = kwargs.get('indices', [])
            indices = [i for i in indices if 0 <= i < len(df)]
            subset = df.iloc[indices]
        else:
            subset = df.head(n_rows)

        # Convert to list of dicts, rounding floats for readability
        rows = []
        for _, row in subset.iterrows():
            d = {}
            for col in subset.columns:
                val = row[col]
                if hasattr(val, 'item'):
                    val = val.item()
                if isinstance(val, float):
                    val = round(val, 6)
                d[col] = val
            rows.append(d)

        return {
            'status': 'success',
            'name': kwargs['data'],
            'shape': [len(df), len(df.columns)],
            'columns': list(subset.columns),
            'n_rows_returned': len(rows),
            'rows': rows,
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }


def _detect_install_method() -> Dict[str, Any]:
    """Inspect sys.prefix / sys.executable to guess how tuiml was installed."""
    import sys
    # Don't resolve() — that follows the python symlink out of the tool venv.
    prefix = sys.prefix.replace("\\", "/")
    exe = sys.executable.replace("\\", "/")

    # Editable install wins over path-matching: check first so a dev checkout
    # imported into any venv is still reported as editable-dev.
    try:
        import tuiml as _pkg
        pkg_dir = Path(_pkg.__file__).resolve().parent
        if (pkg_dir.parent / "pyproject.toml").exists():
            return {"method": "editable-dev", "writable": False,
                    "upgrade_hint": "cd <checkout> && git pull"}
    except Exception:
        pass

    # uv tool install puts the venv under .../uv/tools/tuiml/
    if "/uv/tools/tuiml" in prefix or "/uv/tools/tuiml" in exe:
        return {"method": "uv-tool", "writable": True,
                "upgrade_hint": "uv tool install --reinstall --force tuiml"}

    # Default: assume pip / uv pip
    return {"method": "pip", "writable": True,
            "upgrade_hint": f"{sys.executable} -m pip install --upgrade tuiml"}


def _query_latest_pypi_version(package: str = "tuiml", timeout: float = 5.0) -> Dict[str, Any]:
    """Look up the latest released version of a package on PyPI."""
    import json as _json
    import urllib.request
    url = f"https://pypi.org/pypi/{package}/json"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
        return {"ok": True, "version": data["info"]["version"],
                "released": data["releases"].get(data["info"]["version"], [{}])[0].get("upload_time")}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def execute_system_info(**kwargs) -> Dict[str, Any]:
    """Report installation details for the running TuiML install."""
    import sys
    import platform as _plat
    try:
        import tuiml as _pkg
        pkg_dir = Path(_pkg.__file__).resolve().parent
        version = getattr(_pkg, "__version__", "unknown")
    except Exception as e:
        return {"status": "error", "error": f"cannot import tuiml: {e}",
                "error_type": type(e).__name__}

    install = _detect_install_method()
    result: Dict[str, Any] = {
        "status": "success",
        "version": version,
        "install_method": install["method"],
        "upgrade_hint": install["upgrade_hint"],
        "package_path": str(pkg_dir),
        "site_packages": str(pkg_dir.parent),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": _plat.platform(),
    }

    if kwargs.get("check_latest", True):
        pypi = _query_latest_pypi_version()
        if pypi["ok"]:
            latest = pypi["version"]
            result["latest_version"] = latest
            result["update_available"] = (latest != version)
            if pypi.get("released"):
                result["latest_released"] = pypi["released"]
        else:
            result["latest_version_error"] = pypi["error"]

    return result


def execute_self_update(**kwargs) -> Dict[str, Any]:
    """Upgrade tuiml to the latest PyPI version using the detected installer."""
    import subprocess
    import sys

    install = _detect_install_method()
    method = install["method"]

    if method == "editable-dev":
        return {
            "status": "error",
            "error": "refusing to upgrade an editable / dev checkout — run `git pull` in the source tree instead",
            "error_type": "EditableInstallError",
            "install_method": method,
        }

    target = kwargs.get("target_version")
    spec = f"tuiml=={target}" if target else "tuiml"

    if method == "uv-tool":
        cmd = ["uv", "tool", "install", "--reinstall", "--force", spec]
    else:  # pip
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade", spec]

    if kwargs.get("dry_run"):
        return {
            "status": "success",
            "dry_run": True,
            "install_method": method,
            "command": cmd,
            "note": "no changes made; set dry_run=false to actually run",
        }

    try:
        before = execute_system_info(check_latest=False).get("version")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": "upgrade timed out after 300s",
                "error_type": "TimeoutExpired", "command": cmd}
    except FileNotFoundError as e:
        return {"status": "error",
                "error": f"installer not found on PATH: {e.filename or str(e)}",
                "error_type": "FileNotFoundError", "command": cmd}

    # After upgrade the on-disk package has changed, but the running process is
    # still holding the old module objects. Re-reading the installed version
    # requires a subprocess.
    try:
        probe = subprocess.run(
            [sys.executable, "-c",
             "import importlib, importlib.metadata as m; "
             "importlib.invalidate_caches(); "
             "print(m.version('tuiml'))"],
            capture_output=True, text=True, timeout=15,
        )
        after = probe.stdout.strip() or None
    except Exception:
        after = None

    ok = (proc.returncode == 0)
    return {
        "status": "success" if ok else "error",
        "install_method": method,
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
        "version_before": before,
        "version_after": after,
        "restart_required": ok,
        "note": "Restart the MCP client (Claude Desktop, Cursor, etc.) to load the new version.",
    }


# =============================================================================
# Tool Registry
# =============================================================================

TOOL_EXECUTORS = {
    "tuiml_train": execute_train,
    "tuiml_predict": execute_predict,
    "tuiml_evaluate": execute_evaluate,
    "tuiml_experiment": execute_experiment,
    "tuiml_upload_data": execute_upload_data,
    "tuiml_save_model": execute_save_model,
    "tuiml_serve_model": execute_serve_model,
    "tuiml_stop_server": execute_stop_server,
    "tuiml_server_status": execute_server_status,
    "tuiml_list": execute_list,
    "tuiml_describe": execute_describe,
    "tuiml_search": execute_search,
    "tuiml_plot": execute_plot,
    "tuiml_data_profile": execute_data_profile,
    "tuiml_generate_data": execute_generate_data,
    "tuiml_preprocess": execute_preprocess,
    "tuiml_select_features": execute_select_features,
    "tuiml_statistical_test": execute_statistical_test,
    "tuiml_tune": execute_tune,
    "tuiml_read_data": execute_read_data,
    "tuiml_system_info": execute_system_info,
    "tuiml_self_update": execute_self_update,
}

def get_tool_output_schema(tool_name: str) -> Dict[str, Any]:
    """Get output schema for a tool."""
    return OUTPUT_SCHEMAS.get(tool_name, COMPONENT_OUTPUT_SCHEMA)

def get_tool_annotations(tool_name: str) -> Dict[str, bool]:
    """Get MCP annotations for a tool."""
    # Define annotations for each tool
    TOOL_ANNOTATIONS = {
        "tuiml_train": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True
        },
        "tuiml_predict": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False
        },
        "tuiml_evaluate": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False
        },
        "tuiml_experiment": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True
        },
        "tuiml_upload_data": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False
        },
        "tuiml_save_model": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False
        },
        "tuiml_serve_model": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True
        },
        "tuiml_stop_server": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False
        },
        "tuiml_server_status": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False
        },
        "tuiml_list": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False
        },
        "tuiml_describe": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True
        },
        "tuiml_search": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False
        },
        "tuiml_plot": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False
        },
        "tuiml_data_profile": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False
        },
        "tuiml_generate_data": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False
        },
        "tuiml_preprocess": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False
        },
        "tuiml_select_features": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False
        },
        "tuiml_statistical_test": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False
        },
        "tuiml_tune": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True
        },
        "tuiml_read_data": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False
        }
    }

    # Default annotations for component tools
    DEFAULT_COMPONENT_ANNOTATIONS = {
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }

    return TOOL_ANNOTATIONS.get(tool_name, DEFAULT_COMPONENT_ANNOTATIONS)

def get_workflow_tools() -> Dict[str, Dict]:
    """Get all workflow tool schemas."""
    return {**WORKFLOW_TOOLS, **DISCOVERY_TOOLS}

def execute_tool(tool_name: str, **kwargs) -> Dict[str, Any]:
    """Execute a tool by name."""
    # Check workflow tools first
    if tool_name in TOOL_EXECUTORS:
        return TOOL_EXECUTORS[tool_name](**kwargs)

    # For any component tool, ensure full registry is loaded
    from tuiml.llm.registry import get_tool
    tool = get_tool(tool_name)
    if tool:
        try:
            result = tool.executor(kwargs)
            return {
                'status': 'success',
                'result': str(result),
                'type': result.__class__.__name__
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

    return {'status': 'error', 'error': f"Unknown tool: {tool_name}"}
