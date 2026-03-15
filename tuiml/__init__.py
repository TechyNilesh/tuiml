"""
TuiML - Modern machine learning toolkit inspired by WEKA.

A Python-based ML framework with a plugin architecture for extensibility.

Three levels of API:
    1. High-Level (Functional): tuiml.train() - One-liner with all parameters
    2. Mid-Level (Workflow): tuiml.Workflow() - Fluent chainable interface
    3. Low-Level (OOP): Direct class imports - Maximum control
"""

from tuiml.hub import registry, ComponentType

# High-level API
from tuiml.api import (
    train,
    run,
    predict,
    evaluate,
    experiment,
    save,
    load,
    list_algorithms,
    describe_algorithm,
    search_algorithms,
    serve,
    stop_server,
    server_status,
    PRESETS,
)

# Mid-level API
from tuiml.workflow import Workflow, WorkflowResult

__version__ = "0.1.1"

__all__ = [
    # Core registry
    "hub",
    "ComponentType",

    # High-level API (one-liner functions)
    "train",
    "run",
    "predict",
    "evaluate",
    "experiment",
    "save",
    "load",
    "list_algorithms",
    "describe_algorithm",
    "search_algorithms",
    "PRESETS",
    "serve",
    "stop_server",
    "server_status",

    # Mid-level API (fluent workflow)
    "Workflow",
    "WorkflowResult",
]
