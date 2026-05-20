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


_ALIAS_RE = re.compile(r"^(?P<class>[A-Za-z_][A-Za-z0-9_]*)_v(?P<v>\d+_\d+_\d+)$")


def _resolve_user_algorithm(name: str) -> Optional[Dict[str, Any]]:
    """Map an experiment algorithm name back to a user-algorithm directory.

    Accepts both the bare class name (``WeightedTreeBag`` — resolves to the
    newest on-disk version) and a pinned alias (``WeightedTreeBag_v1_0_1``).
    Returns a dict ``{name, version, dir}`` or ``None`` if this name doesn't
    correspond to a user algorithm.
    """
    # Pinned alias first.
    m = _ALIAS_RE.match(name)
    if m:
        base = m.group("class")
        ver = m.group("v").replace("_", ".")
        p = USER_ALGS_DIR / base / ver
        if p.is_dir():
            return {"name": base, "version": ver, "dir": p}

    # Bare class name: return the highest semver present on disk.
    base_dir = USER_ALGS_DIR / name
    if base_dir.is_dir():
        versions = sorted(
            (v for v in base_dir.iterdir() if v.is_dir() and _VERSION_RE.match(v.name)),
            key=lambda p: tuple(int(x) for x in p.name.split(".")),
            reverse=True,
        )
        if versions:
            v = versions[0]
            return {"name": name, "version": v.name, "dir": v}
    return None


def _best_metric(scores_block: Dict[str, Any]) -> Optional[Tuple[str, float]]:
    """Pick a representative metric: prefer accuracy_score, then first available."""
    if not isinstance(scores_block, dict):
        return None
    if "accuracy_score" in scores_block and isinstance(scores_block["accuracy_score"], dict):
        mean = scores_block["accuracy_score"].get("mean")
        if mean is not None:
            return "accuracy_score", float(mean)
    for metric, block in scores_block.items():
        if isinstance(block, dict) and block.get("mean") is not None:
            return metric, float(block["mean"])
    return None


def record_experiment_runs(experiment_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Scan a ``tuiml_experiment`` result and append entries to the matching
    user algorithms' ``runs.jsonl`` files.

    Returns the list of appended entries (empty if none matched or the feature
    is disabled). Never raises — logging failures are swallowed so the caller's
    experiment result is not affected.
    """
    import datetime as _dt, json as _json

    if not isinstance(experiment_result, dict) or experiment_result.get("status") != "success":
        return []
    results = experiment_result.get("results")
    if not isinstance(results, dict):
        return []

    appended: List[Dict[str, Any]] = []
    ts = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    cv = experiment_result.get("cv") or experiment_result.get("n_folds")

    for dataset_name, algo_scores in results.items():
        if not isinstance(algo_scores, dict):
            continue
        for algo_name, scores_block in algo_scores.items():
            match = _resolve_user_algorithm(algo_name)
            if not match:
                continue
            entry: Dict[str, Any] = {
                "timestamp": ts,
                "dataset": dataset_name,
                "cv": cv,
                "algorithm_name": algo_name,
                "resolved_version": match["version"],
                "metrics": scores_block,
            }
            best = _best_metric(scores_block)
            if best:
                entry["primary_metric"] = best[0]
                entry["primary_score"] = best[1]
            log_path = match["dir"] / "runs.jsonl"
            try:
                with log_path.open("a", encoding="utf-8") as fh:
                    fh.write(_json.dumps(entry) + "\n")
                appended.append({"name": match["name"], "version": match["version"],
                                 "dataset": dataset_name, "path": str(log_path)})
            except Exception:
                pass
    return appended


def research_log(name: Optional[str] = None) -> Dict[str, Any]:
    """Return the aggregated research view for one (or all) user algorithms.

    Combines ``metadata.json`` (static: class, kind, version, source hash)
    with ``runs.jsonl`` (dynamic: every recorded experiment run) and computes:
    best primary score per version, run count per version, and the most recent
    timestamp. Agents use this as the equivalent of the landing-page "research
    log" panel.
    """
    import json as _json

    if not USER_ALGS_DIR.exists():
        return {"status": "success", "algorithms": [], "root": str(USER_ALGS_DIR)}

    roots: List[Path]
    if name:
        base = USER_ALGS_DIR / name
        if not base.is_dir():
            return {"status": "error", "error_type": "NotFound",
                    "error": f"no user algorithm named {name!r}"}
        roots = [base]
    else:
        roots = [p for p in USER_ALGS_DIR.iterdir() if p.is_dir()]

    algorithms: List[Dict[str, Any]] = []
    for name_dir in sorted(roots):
        versions: List[Dict[str, Any]] = []
        for ver_dir in sorted(
            (p for p in name_dir.iterdir() if p.is_dir() and _VERSION_RE.match(p.name)),
            key=lambda p: tuple(int(x) for x in p.name.split(".")),
        ):
            meta = _read_metadata(ver_dir)
            runs_path = ver_dir / "runs.jsonl"
            runs: List[Dict[str, Any]] = []
            if runs_path.exists():
                try:
                    for line in runs_path.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            runs.append(_json.loads(line))
                        except Exception:
                            continue
                except Exception:
                    runs = []

            best_score: Optional[float] = None
            best_dataset: Optional[str] = None
            best_metric: Optional[str] = None
            for r in runs:
                score = r.get("primary_score")
                if score is None:
                    continue
                if best_score is None or score > best_score:
                    best_score = float(score)
                    best_dataset = r.get("dataset")
                    best_metric = r.get("primary_metric")

            last_run_ts = max((r.get("timestamp", "") for r in runs), default=None) or None

            versions.append({
                "version": ver_dir.name,
                "class_name": meta.get("class_name", name_dir.name),
                "kind": meta.get("kind"),
                "description": meta.get("description"),
                "source_hash": meta.get("source_hash"),
                "pinned_alias": _versioned_alias_name(
                    meta.get("class_name", name_dir.name), ver_dir.name),
                "run_count": len(runs),
                "best_score": best_score,
                "best_on_dataset": best_dataset,
                "best_metric": best_metric,
                "last_run": last_run_ts,
                "path": str(ver_dir),
            })

        # Which version is the current best across all recorded runs?
        scored = [v for v in versions if v["best_score"] is not None]
        overall_best = max(scored, key=lambda v: v["best_score"])["version"] if scored else None

        algorithms.append({
            "name": name_dir.name,
            "versions": versions,
            "overall_best_version": overall_best,
        })

    return {
        "status": "success",
        "count": len(algorithms),
        "algorithms": algorithms,
        "root": str(USER_ALGS_DIR),
    }


def read_source(name: str, version: Optional[str] = None,
                builtin: bool = False) -> Dict[str, Any]:
    """Return the full source code of a user or built-in algorithm.

    For user algorithms, ``name`` is the directory name under USER_ALGS_DIR.
    For built-in algorithms, set ``builtin=True`` and pass the class name or
    file stem (e.g. ``'RandomForestClassifier'`` or ``'random_forest'``).
    """
    if builtin:
        return _read_builtin_source(name)

    resolved = _resolve_user_algorithm(name) if version is None else None
    if resolved is None and version is not None:
        err = _validate_name(name) or _validate_version(version)
        if err:
            return {"status": "error", "error_type": "ValueError", "error": err}
        resolved = {"name": name, "version": version, "dir": USER_ALGS_DIR / name / version}
    if resolved is None:
        return {"status": "error", "error_type": "NotFound",
                "error": f"no user algorithm named {name!r} found on disk"}

    alg_file = resolved["dir"] / "algorithm.py"
    if not alg_file.exists():
        return {"status": "error", "error_type": "NotFound",
                "error": f"algorithm.py not found at {alg_file}"}

    source = alg_file.read_text(encoding="utf-8")
    meta = _read_metadata(resolved["dir"])
    lines = source.splitlines()
    numbered = "\n".join(f"{i+1:4d} | {line}" for i, line in enumerate(lines))
    return {
        "status": "success",
        "name": name,
        "version": resolved["version"],
        "class_name": meta.get("class_name", name),
        "kind": meta.get("kind"),
        "path": str(alg_file),
        "line_count": len(lines),
        "source": source,
        "source_with_line_numbers": numbered,
    }


def _read_builtin_source(name: str) -> Dict[str, Any]:
    """Locate and return source for a built-in tuiml algorithm file."""
    import tuiml.algorithms as _alg_pkg
    alg_root = Path(_alg_pkg.__file__).parent

    # Try class name → search all .py files for class definition
    candidates: List[Path] = []
    stem = name.lower().replace(" ", "_")
    for py_file in alg_root.rglob("*.py"):
        if py_file.name.startswith("_"):
            continue
        if py_file.stem == stem or py_file.stem == name:
            candidates.append(py_file)
        elif name in py_file.read_text(encoding="utf-8", errors="ignore"):
            # class name appears in the file
            text = py_file.read_text(encoding="utf-8", errors="ignore")
            if f"class {name}(" in text:
                candidates.insert(0, py_file)
            else:
                candidates.append(py_file)

    if not candidates:
        return {"status": "error", "error_type": "NotFound",
                "error": f"no built-in algorithm file found for {name!r}. "
                         "Use tuiml_list_algorithm_files to see all built-in paths."}

    py_file = candidates[0]
    source = py_file.read_text(encoding="utf-8")
    lines = source.splitlines()
    numbered = "\n".join(f"{i+1:4d} | {line}" for i, line in enumerate(lines))
    return {
        "status": "success",
        "name": name,
        "builtin": True,
        "path": str(py_file),
        "line_count": len(lines),
        "source": source,
        "source_with_line_numbers": numbered,
        "note": "Built-in algorithms are read-only. Fork to user algorithms with tuiml_create_algorithm.",
    }


def list_algorithm_files(builtin: bool = True, user: bool = True) -> Dict[str, Any]:
    """List all algorithm source files — built-in and/or user-authored."""
    results: List[Dict[str, Any]] = []

    if builtin:
        import tuiml.algorithms as _alg_pkg
        alg_root = Path(_alg_pkg.__file__).parent
        for py_file in sorted(alg_root.rglob("*.py")):
            if py_file.name.startswith("_"):
                continue
            rel = py_file.relative_to(alg_root)
            results.append({
                "type": "builtin",
                "path": str(py_file),
                "relative_path": str(rel),
                "category": rel.parts[0] if len(rel.parts) > 1 else "root",
                "file": py_file.name,
            })

    if user and USER_ALGS_DIR.exists():
        for name_dir in sorted(USER_ALGS_DIR.iterdir()):
            if not name_dir.is_dir():
                continue
            for ver_dir in sorted(name_dir.iterdir()):
                alg_file = ver_dir / "algorithm.py"
                if alg_file.exists():
                    meta = _read_metadata(ver_dir)
                    results.append({
                        "type": "user",
                        "name": name_dir.name,
                        "version": ver_dir.name,
                        "class_name": meta.get("class_name", name_dir.name),
                        "kind": meta.get("kind"),
                        "path": str(alg_file),
                    })

    return {"status": "success", "count": len(results), "files": results}


def search_source(query: str, name: Optional[str] = None,
                  builtin: bool = True, user: bool = True) -> Dict[str, Any]:
    """Grep for ``query`` across algorithm source files.

    Pass ``name`` to scope the search to one user algorithm. Returns matching
    lines with file path and line number — same pattern as grep.
    """
    import re as _re
    try:
        pattern = _re.compile(query)
    except _re.error as e:
        return {"status": "error", "error_type": "ValueError",
                "error": f"invalid regex: {e}"}

    files_info = list_algorithm_files(builtin=builtin, user=user)
    search_files: List[Path] = []

    for f in files_info["files"]:
        p = Path(f["path"])
        if name and f.get("type") == "user" and f.get("name") != name:
            continue
        search_files.append(p)

    matches: List[Dict[str, Any]] = []
    for py_file in search_files:
        try:
            lines = py_file.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        for i, line in enumerate(lines):
            if pattern.search(line):
                matches.append({
                    "path": str(py_file),
                    "line_number": i + 1,
                    "line": line,
                })

    return {
        "status": "success",
        "query": query,
        "match_count": len(matches),
        "matches": matches,
    }


def edit_algorithm(name: str, old_string: str, new_string: str,
                   version: Optional[str] = None,
                   bump_version: bool = False) -> Dict[str, Any]:
    """Apply a str_replace edit to a user algorithm source.

    Replaces exactly one occurrence of ``old_string`` with ``new_string``.
    Fails if ``old_string`` is not found, or if it appears more than once
    (ambiguous — make the search string more specific in that case).

    After the edit the source is AST-validated and re-registered. Optionally
    bumps the patch version and saves as a new file.
    """
    err = _validate_name(name)
    if err:
        return {"status": "error", "error_type": "ValueError", "error": err}

    resolved = _resolve_user_algorithm(name) if version is None else None
    if resolved is None and version is not None:
        ver_err = _validate_version(version)
        if ver_err:
            return {"status": "error", "error_type": "ValueError", "error": ver_err}
        resolved = {"name": name, "version": version, "dir": USER_ALGS_DIR / name / version}
    if resolved is None:
        return {"status": "error", "error_type": "NotFound",
                "error": f"no user algorithm named {name!r} found on disk"}

    alg_file = resolved["dir"] / "algorithm.py"
    if not alg_file.exists():
        return {"status": "error", "error_type": "NotFound",
                "error": f"algorithm.py not found at {alg_file}"}

    source = alg_file.read_text(encoding="utf-8")

    # Uniqueness check — the heart of the str_replace approach.
    count = source.count(old_string)
    if count == 0:
        return {
            "status": "error", "error_type": "NotFound",
            "error": "old_string not found in the source. Read the file first to verify the exact text.",
        }
    if count > 1:
        return {
            "status": "error", "error_type": "Ambiguous",
            "error": f"old_string appears {count} times — make it more specific by including more surrounding context.",
        }

    new_source = source.replace(old_string, new_string, 1)

    # AST validate the edited source.
    ok, reason = _ast_validate(new_source)
    if not ok:
        return {"status": "error", "error_type": "UnsafeSource", "error": reason}

    current_version = resolved["version"]
    if bump_version:
        parts = current_version.split(".")
        new_ver = f"{parts[0]}.{parts[1]}.{int(parts[2]) + 1}"
        target_dir = USER_ALGS_DIR / name / new_ver
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / "algorithm.py"
        save_version = new_ver
    else:
        target_file = alg_file
        save_version = current_version

    target_file.write_text(new_source, encoding="utf-8")

    try:
        cls, detected_kind = _import_and_register(target_file, name, save_version)
    except Exception as e:
        return {"status": "error", "error_type": "LoadError", "error": str(e)}

    meta = _read_metadata(resolved["dir"])
    new_meta = {
        **meta,
        "version": save_version,
        "source_hash": _source_hash(new_source),
    }
    _write_metadata(target_file.parent, new_meta)

    return {
        "status": "success",
        "name": name,
        "version": save_version,
        "previous_version": current_version,
        "class_name": cls.__name__,
        "path": str(target_file),
        "source_hash": new_meta["source_hash"],
        "registered_as": [cls.__name__, _versioned_alias_name(cls.__name__, save_version)],
        "note": "Algorithm re-registered. Run tuiml_experiment to validate the change.",
    }


def load_all() -> Dict[str, Any]:
    """Scan ``USER_ALGS_DIR`` and register every algorithm found.

    Called once at MCP server startup. Failures on individual files are logged
    but do not abort the whole load.
    """
    if not USER_ALGS_DIR.exists():
        return {"status": "success", "loaded": 0, "errors": []}

    loaded: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for name_dir in sorted(USER_ALGS_DIR.iterdir()):
        if not name_dir.is_dir():
            continue
        ver_dirs = sorted(
            (d for d in name_dir.iterdir() if d.is_dir() and (d / "algorithm.py").exists()),
            key=lambda d: d.name,
        )
        if not ver_dirs:
            continue
        ver_dir = ver_dirs[-1]  # only load the latest version
        alg_file = ver_dir / "algorithm.py"
        try:
            cls, kind = _import_and_register(alg_file, name_dir.name, ver_dir.name)
            loaded.append({"name": cls.__name__, "kind": kind, "version": ver_dir.name})
        except Exception as e:  # keep going on partial failures
            errors.append({"path": str(alg_file), "error": str(e)})

    return {"status": "success", "loaded": len(loaded),
            "algorithms": loaded, "errors": errors}
