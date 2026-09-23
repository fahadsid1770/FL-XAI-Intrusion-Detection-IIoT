"""Model persistence via joblib.

Trained estimators (and the calibrated hybrid) are serialized to
``config.MODELS_DIR`` together with a small JSON manifest that records the
exact display name of each model, so a notebook/script can reload everything
without retraining — and without losing the original model names.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib

from . import config

MANIFEST_FILENAME = "manifest.json"


def _safe_name(name: str) -> str:
    """Convert a display name to a filesystem-safe slug."""
    return (
        name.replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
        .lower()
    )


def save_model(model, name: str, models_dir=None) -> str:
    """Serialize a single estimator and return its path."""
    models_dir = Path(models_dir or config.MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{_safe_name(name)}.joblib"
    joblib.dump(model, path)
    return str(path)


def load_model(name: str, models_dir=None):
    """Load a single estimator by its exact display name."""
    models_dir = Path(models_dir or config.MODELS_DIR)
    return joblib.load(models_dir / f"{_safe_name(name)}.joblib")


def save_all_models(trained: dict, calibrated_hybrid=None, models_dir=None) -> dict:
    """Persist every trained model plus the calibrated hybrid.

    Writes a ``manifest.json`` mapping each saved file to its exact display
    name, and returns a ``{display_name: path}`` mapping.
    """
    models_dir = Path(models_dir or config.MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"models": [], "calibrated": None}
    paths = {}

    for name, model in trained.items():
        path = save_model(model, name, models_dir)
        paths[name] = path
        manifest["models"].append({"name": name, "file": Path(path).name})

    if calibrated_hybrid is not None:
        cal_name = "Hybrid XGB-LGBM (calibrated)"
        path = save_model(calibrated_hybrid, cal_name, models_dir)
        paths[cal_name] = path
        manifest["calibrated"] = {"name": cal_name, "file": Path(path).name}

    (models_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2))
    return paths


def load_all_models(models_dir=None):
    """Reload every model saved by :func:`save_all_models`.

    Returns ``(trained, calibrated_hybrid)`` with exact display-name keys.
    Falls back to a best-effort glob scan if the manifest is missing.
    """
    models_dir = Path(models_dir or config.MODELS_DIR)
    manifest_path = models_dir / MANIFEST_FILENAME

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        trained = {
            entry["name"]: joblib.load(models_dir / entry["file"])
            for entry in manifest.get("models", [])
        }
        calibrated = None
        if manifest.get("calibrated"):
            calibrated = joblib.load(models_dir / manifest["calibrated"]["file"])
        return trained, calibrated

    # Backward-compatible fallback: infer names from filenames.
    trained = {}
    calibrated = None
    for path in sorted(models_dir.glob("*.joblib")):
        model = joblib.load(path)
        if "calibrated" in path.stem:
            calibrated = model
        else:
            trained[path.stem.replace("_", " ").title()] = model
    return trained, calibrated
