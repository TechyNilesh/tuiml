"""Agent-authored algorithms — persistence, safety checks, and registry bootstrap.

Agents can call ``tuiml_create_algorithm`` with raw Python source describing a
new ``@classifier`` / ``@regressor`` class. This module:

1. AST-validates the source against a conservative denylist (no network,
   subprocess, or filesystem escape hatches).
2. Saves each accepted submission under
   ``~/.tuiml/user_algorithms/<Name>/<version>/algorithm.py``.
3. Imports the file and registers both the *versioned* alias
   (``MyAlg_v1_0_0``) and the bare latest-alias (``MyAlg``) into the TuiML
   registry, so every existing MCP tool (``tuiml_train``, ``tuiml_experiment``,
   ``tuiml_describe``) works on user algorithms unchanged.
4. On ``load_all()`` scans the directory and re-registers everything,
   preserving agent work across MCP server restarts.

Gated by the ``TUIML_ALLOW_USER_ALGORITHMS`` environment variable. When unset
or ``0``, every write tool returns an error explaining how to enable the
feature; read tools still work. Set ``TUIML_ALLOW_USER_ALGORITHMS=1`` to opt in.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

USER_ALGS_DIR = Path.home() / ".tuiml" / "user_algorithms"
ENABLE_ENV = "TUIML_ALLOW_USER_ALGORITHMS"


# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------

def is_enabled() -> bool:
    """Return True if agent-authored algorithms are allowed on this machine."""
    return os.environ.get(ENABLE_ENV, "0").lower() in {"1", "true", "yes", "on"}


def _disabled_error() -> Dict[str, Any]:
    return {
        "status": "error",
        "error_type": "FeatureDisabled",
        "error": (
            "Agent-authored algorithms are disabled on this machine. "
            f"Set environment variable {ENABLE_ENV}=1 to enable. "
            "User code runs in-process and is only AST-filtered, not sandboxed."
        ),
    }


# ---------------------------------------------------------------------------
# Code templates (skeletons)
# ---------------------------------------------------------------------------

_CLASSIFIER_TEMPLATE = '''"""{description}"""

import numpy as np
from typing import Dict, Any

from tuiml.base.algorithms import Classifier, classifier


@classifier(tags=["custom"], version="{version}")
class {class_name}(Classifier):
    """{description}

    Parameters
    ----------
    n_neighbors : int, default=5
        Placeholder hyperparameter — replace with your own.
    """

    def __init__(self, n_neighbors: int = 5):
        super().__init__()
        self.n_neighbors = n_neighbors

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {{
            "n_neighbors": {{"type": int, "default": 5, "range": (1, 100)}},
        }}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "{class_name}":
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        # TODO: implement training logic
        self._most_common_ = self.classes_[np.argmax(np.bincount(y.astype(int)))] if self.n_classes_ else 0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        # TODO: implement prediction logic
        return np.full(len(X), self._most_common_)
'''

_REGRESSOR_TEMPLATE = '''"""{description}"""

import numpy as np
from typing import Dict, Any

from tuiml.base.algorithms import Regressor, regressor


@regressor(tags=["custom"], version="{version}")
class {class_name}(Regressor):
    """{description}

    Parameters
    ----------
    alpha : float, default=1.0
        Placeholder hyperparameter — replace with your own.
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {{
            "alpha": {{"type": float, "default": 1.0, "range": (0.0, 10.0)}},
        }}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "{class_name}":
        X = np.asarray(X)
        y = np.asarray(y, dtype=float)
        # TODO: implement training logic
        self.mean_ = float(y.mean())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        # TODO: implement prediction logic
        return np.full(len(X), self.mean_)
'''


def skeleton(kind: str, class_name: str = "MyAlgorithm",
             version: str = "1.0.0",
             description: str = "Describe what your algorithm does.") -> Dict[str, Any]:
    """Return a ready-to-fill template for a new algorithm."""
    kind = kind.lower()
    if kind not in {"classifier", "regressor"}:
        return {
            "status": "error",
            "error_type": "ValueError",
            "error": f"kind must be 'classifier' or 'regressor', got {kind!r}",
        }
    template = _CLASSIFIER_TEMPLATE if kind == "classifier" else _REGRESSOR_TEMPLATE
    return {
        "status": "success",
        "kind": kind,
        "class_name": class_name,
        "version": version,
        "code": template.format(
            class_name=class_name,
            version=version,
            description=description,
        ),
        "notes": (
            "Fill in fit() and predict(), adjust __init__ hyperparameters and "
            "get_parameter_schema(), then pass the completed source to "
            "tuiml_create_algorithm."
        ),
    }


# ---------------------------------------------------------------------------
# AST safety check
# ---------------------------------------------------------------------------

# Top-level modules we refuse to import from user code. Agents building ML
# algorithms don't need any of these and they're the usual escape hatches.
_FORBIDDEN_MODULES = {
    "subprocess", "socket", "os", "shutil", "urllib", "requests", "httpx",
    "http", "ftplib", "smtplib", "paramiko", "telnetlib", "ctypes", "webbrowser",
    "pty", "asyncio.subprocess",
}

# Names agents shouldn't call. `open` is technically fine but pushes users
# towards reading files they shouldn't.
_FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__", "open", "input"}

# Even attribute accesses like os.system should fail — but our module denylist
# already blocks `import os`.
_FORBIDDEN_ATTRS = {"system", "popen", "spawn", "spawnl", "spawnv"}


def _ast_validate(source: str) -> Tuple[bool, str]:
    """Walk the AST and reject source that uses denylisted modules / calls.

    Returns ``(ok, reason)``. ``reason`` is empty on success.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return False, f"syntax error: {e}"

    # Require at least one decorated Classifier / Regressor class
    found_decorator = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _FORBIDDEN_MODULES:
                    return False, f"forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root in _FORBIDDEN_MODULES:
                    return False, f"forbidden import from: {node.module}"
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in _FORBIDDEN_CALLS:
                return False, f"forbidden call: {f.id}()"
            if isinstance(f, ast.Attribute) and f.attr in _FORBIDDEN_ATTRS:
                return False, f"forbidden attribute call: .{f.attr}()"
        elif isinstance(node, ast.ClassDef):
            for deco in node.decorator_list:
                name = None
                if isinstance(deco, ast.Call) and isinstance(deco.func, ast.Name):
                    name = deco.func.id
                elif isinstance(deco, ast.Name):
                    name = deco.id
                if name in {"classifier", "regressor", "clusterer", "associator"}:
                    found_decorator = True

    if not found_decorator:
        return False, (
            "no @classifier / @regressor decorated class found — did you "
            "forget the decorator? Call tuiml_algorithm_skeleton for a template."
        )

    return True, ""


# ---------------------------------------------------------------------------
# Disk layout helpers
# ---------------------------------------------------------------------------

_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


def _validate_name(name: str) -> Optional[str]:
    if not _NAME_RE.match(name):
        return f"name must be a valid Python identifier, got {name!r}"
    return None


def _validate_version(version: str) -> Optional[str]:
    if not _VERSION_RE.match(version):
        return f"version must be MAJOR.MINOR.PATCH semver, got {version!r}"
    return None


def _alg_dir(name: str, version: str) -> Path:
    return USER_ALGS_DIR / name / version


def _algorithm_file(name: str, version: str) -> Path:
    return _alg_dir(name, version) / "algorithm.py"


def _source_hash(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Load + register
# ---------------------------------------------------------------------------

def _versioned_alias_name(name: str, version: str) -> str:
    """Return a valid Python identifier encoding the version, e.g. MyGBM_v1_0_0."""
    return f"{name}_v{version.replace('.', '_')}"


def _import_and_register(file_path: Path, name: str, version: str) -> Tuple[Any, str]:
    """Import ``algorithm.py`` and register both a versioned alias and a latest alias.

    Returns ``(class_obj, kind)`` where kind is ``'classifier'`` / ``'regressor'``.
    Raises ``RuntimeError`` on failure.
    """
    module_name = f"_tuiml_user_{name}_v{version.replace('.', '_')}"
    # Remove any previously-imported copy so decorators re-fire.
    sys.modules.pop(module_name, None)

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not build import spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        sys.modules.pop(module_name, None)
        raise RuntimeError(f"error while importing user algorithm: {e}") from e

    # Find the decorated class in the imported module.
    from tuiml.base.algorithms import Classifier, Regressor, classifier, regressor
    target_cls = None
    kind = None
    for obj in vars(module).values():
        if not inspect.isclass(obj) or obj.__module__ != module_name:
            continue
        if issubclass(obj, Classifier) and obj is not Classifier:
            target_cls, kind = obj, "classifier"
            break
        if issubclass(obj, Regressor) and obj is not Regressor:
            target_cls, kind = obj, "regressor"
            break
    if target_cls is None:
        raise RuntimeError("imported module defines no Classifier/Regressor subclass")

    # Register a versioned alias so A/B comparisons between versions work.
    alias_name = _versioned_alias_name(target_cls.__name__, version)
    if alias_name != target_cls.__name__:
        alias_cls = type(alias_name, (target_cls,), {
            "__doc__": target_cls.__doc__,
            "__module__": module_name,
        })
        decorator = classifier if kind == "classifier" else regressor
        # Re-apply the decorator to register the alias. We pass empty tags so
        # the decorator doesn't fail looking for metadata.
        decorator(tags=["custom", f"version={version}"], version=version)(alias_cls)

    return target_cls, kind


def _write_metadata(dir_path: Path, metadata: Dict[str, Any]) -> None:
    import json
    (dir_path / "metadata.json").write_text(json.dumps(metadata, indent=2))


def _read_metadata(dir_path: Path) -> Dict[str, Any]:
    import json
    path = dir_path / "metadata.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Public API used by MCP tool executors
# ---------------------------------------------------------------------------

def create(name: str, kind: str, code: str,
           version: str = "1.0.0",
           description: Optional[str] = None,
           force: bool = False) -> Dict[str, Any]:
    """Persist, validate, and register a new user algorithm.

    Returns a dict with ``status`` and, on success, the registered aliases and
    source hash.
    """
    if not is_enabled():
        return _disabled_error()

    err = _validate_name(name) or _validate_version(version)
    if err:
        return {"status": "error", "error_type": "ValueError", "error": err}
    kind = kind.lower()
    if kind not in {"classifier", "regressor"}:
        return {"status": "error", "error_type": "ValueError",
                "error": f"kind must be 'classifier' or 'regressor', got {kind!r}"}

    ok, reason = _ast_validate(code)
    if not ok:
        return {"status": "error", "error_type": "UnsafeSource", "error": reason}

    target_file = _algorithm_file(name, version)
    if target_file.exists() and not force:
        return {
            "status": "error", "error_type": "AlreadyExists",
            "error": f"{name} v{version} already exists at {target_file}. "
                     "Pass force=true to overwrite or bump the version.",
            "path": str(target_file),
        }

    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_text(code)

    try:
        cls, detected_kind = _import_and_register(target_file, name, version)
    except Exception as e:
        # Roll back the file write so we don't keep un-loadable code.
        try:
            target_file.unlink(missing_ok=True)
        except Exception:
            pass
        return {"status": "error", "error_type": "LoadError", "error": str(e)}

    if detected_kind != kind:
        return {
            "status": "error", "error_type": "KindMismatch",
            "error": f"declared kind={kind} but imported class is a {detected_kind}",
        }

    metadata = {
        "name": name,
        "class_name": cls.__name__,
        "kind": kind,
        "version": version,
        "description": description or (cls.__doc__ or "").splitlines()[0] if cls.__doc__ else "",
        "source_hash": _source_hash(code),
    }
    _write_metadata(target_file.parent, metadata)

    return {
        "status": "success",
        "registered_as": [cls.__name__, _versioned_alias_name(cls.__name__, version)],
        "name": name,
        "class_name": cls.__name__,
        "kind": kind,
        "version": version,
        "source_hash": metadata["source_hash"],
        "path": str(target_file),
        "usage_hint": (
            f"Train with: tuiml_train(algorithm='{cls.__name__}', ...) "
            f"or the pinned version: tuiml_train(algorithm='{_versioned_alias_name(cls.__name__, version)}', ...)"
        ),
    }


def list_all() -> Dict[str, Any]:
    """List every user-authored algorithm on disk."""
    if not USER_ALGS_DIR.exists():
        return {"status": "success", "algorithms": [], "root": str(USER_ALGS_DIR)}

    rows: List[Dict[str, Any]] = []
    for name_dir in sorted(USER_ALGS_DIR.iterdir()):
        if not name_dir.is_dir():
            continue
        for ver_dir in sorted(name_dir.iterdir()):
            if not ver_dir.is_dir():
                continue
            meta = _read_metadata(ver_dir)
            rows.append({
                "name": meta.get("name", name_dir.name),
                "class_name": meta.get("class_name", name_dir.name),
                "kind": meta.get("kind"),
                "version": meta.get("version", ver_dir.name),
                "description": meta.get("description"),
                "source_hash": meta.get("source_hash"),
                "path": str(ver_dir / "algorithm.py"),
                "versioned_alias": _versioned_alias_name(
                    meta.get("class_name", name_dir.name),
                    meta.get("version", ver_dir.name),
                ),
            })
    return {"status": "success", "algorithms": rows, "count": len(rows)}


def delete(name: str, version: Optional[str] = None) -> Dict[str, Any]:
    """Delete one version (or every version when ``version`` is None)."""
    if not is_enabled():
        return _disabled_error()
    err = _validate_name(name)
    if err:
        return {"status": "error", "error_type": "ValueError", "error": err}

    base = USER_ALGS_DIR / name
    if not base.exists():
        return {"status": "error", "error_type": "NotFound",
                "error": f"no user algorithm named {name!r} on disk"}

    removed: List[str] = []
    if version is None:
        for ver_dir in base.iterdir():
            if ver_dir.is_dir():
                shutil.rmtree(ver_dir)
                removed.append(ver_dir.name)
        shutil.rmtree(base, ignore_errors=True)
    else:
        ver_err = _validate_version(version)
        if ver_err:
            return {"status": "error", "error_type": "ValueError", "error": ver_err}
        ver_dir = base / version
        if not ver_dir.exists():
            return {"status": "error", "error_type": "NotFound",
                    "error": f"{name} v{version} is not installed"}
        shutil.rmtree(ver_dir)
        removed.append(version)
        # Remove empty parent.
        if base.exists() and not any(base.iterdir()):
            base.rmdir()

    return {
        "status": "success",
        "name": name,
        "removed_versions": removed,
        "note": "Registry entries remain until the MCP server restarts.",
    }


def load_all() -> Dict[str, Any]:
    """Scan ``USER_ALGS_DIR`` and register every algorithm found.

    Called once at MCP server startup. Failures on individual files are logged
    but do not abort the whole load.
    """
    if not is_enabled():
        return {"status": "skipped", "reason": f"{ENABLE_ENV} not set"}

    if not USER_ALGS_DIR.exists():
        return {"status": "success", "loaded": 0, "errors": []}

    loaded: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for name_dir in sorted(USER_ALGS_DIR.iterdir()):
        if not name_dir.is_dir():
            continue
        for ver_dir in sorted(name_dir.iterdir()):
            if not ver_dir.is_dir():
                continue
            alg_file = ver_dir / "algorithm.py"
            if not alg_file.exists():
                continue
            try:
                cls, kind = _import_and_register(alg_file, name_dir.name, ver_dir.name)
                loaded.append({"name": cls.__name__, "kind": kind, "version": ver_dir.name})
            except Exception as e:  # keep going on partial failures
                errors.append({"path": str(alg_file), "error": str(e)})

    return {"status": "success", "loaded": len(loaded),
            "algorithms": loaded, "errors": errors}
