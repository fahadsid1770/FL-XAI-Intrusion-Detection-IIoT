"""Production-grade model export.

Exports the trained models into portable, language/runtime-agnostic artifacts
rather than pickle-only ``.joblib`` files:

  * XGBoost native JSON  (``xgboost.json``)
  * LightGBM native text (``lightgbm.txt``)
  * CatBoost native       (``catboost.cbm``)
  * ONNX                  (optional; requires onnx/onnxruntime/skl2onnx/onnxmltools)
  * ensemble spec JSON    (describes the soft-voting combination)
  * metadata manifest JSON (version, SHA-256 hashes, metrics, features, classes)

All artifacts are written to ``outputs/export/``.
"""
from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config

EXPORT_SUBDIR = "export"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _library_versions() -> dict:
    versions = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "sklearn", "xgboost", "lightgbm", "catboost", "onnxruntime"):
        try:
            module = __import__("sklearn" if name == "sklearn" else name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[name] = None
    return versions


# --------------------------------------------------------------------------- #
# Native exports (no extra dependencies)
# --------------------------------------------------------------------------- #
def export_xgboost_native(model, out_dir: Path) -> str:
    """Export a fitted XGBClassifier to native JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "xgboost.json"
    model.get_booster().save_model(str(path))
    return str(path)


def export_lightgbm_native(model, out_dir: Path) -> str:
    """Export a fitted LGBMClassifier to native text."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "lightgbm.txt"
    model.booster_.save_model(str(path))
    return str(path)


def export_catboost_native(model, out_dir: Path) -> str:
    """Export a fitted CatBoostClassifier to native .cbm."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "catboost.cbm"
    model.save_model(str(path), format="cbm")
    return str(path)


def export_random_forest_native(model, out_dir: Path) -> str:
    """RandomForest has no portable native format; fall back to joblib.

    The RF model is a baseline (not the deployment model), so joblib is an
    acceptable artifact here; ONNX is available via :func:`export_onnx`.
    """
    import joblib

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "random_forest.joblib"
    joblib.dump(model, path)
    return str(path)


# --------------------------------------------------------------------------- #
# Ensemble spec
# --------------------------------------------------------------------------- #
def export_ensemble_spec(out_dir: Path, base_specs: list[dict], class_names, feature_names,
                         weights: list[float] | None = None) -> str:
    """Write a JSON spec describing the soft-voting ensemble combination.

    ``base_specs`` is a list of ``{"name", "file", "format"}`` entries; the
    runtime loads each base model, averages per-class probabilities using
    ``weights``, and predicts ``argmax``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(base_specs)
    weights = weights or [1.0 / n] * n
    spec = {
        "type": "soft_voting_ensemble",
        "decision_rule": "argmax(weighted_mean(base_model_probabilities))",
        "weights": weights,
        "base_models": base_specs,
        "class_names": list(class_names),
        "feature_names": list(feature_names),
    }
    path = out_dir / "ensemble_spec.json"
    path.write_text(json.dumps(spec, indent=2))
    return str(path)


# --------------------------------------------------------------------------- #
# Metadata manifest
# --------------------------------------------------------------------------- #
def build_metadata(out_dir: Path, model_entries: list[dict], metrics: dict,
                   class_names, feature_names, random_state: int = config.RANDOM_STATE) -> str:
    """Write ``metadata.json`` recording provenance and integrity hashes."""
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for e in model_entries:
        entry = dict(e)
        entry["sha256"] = _sha256(Path(e["file"]))
        entries.append(entry)

    metadata = {
        "exported_at": _now_iso(),
        "random_state": random_state,
        "dataset": "Edge-IIoTset (ML-EdgeIIoT-dataset.csv)",
        "n_features": len(feature_names),
        "n_classes": len(class_names),
        "class_names": list(class_names),
        "feature_names": list(feature_names),
        "metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in metrics.items()},
        "library_versions": _library_versions(),
        "models": entries,
    }
    path = out_dir / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2))
    return str(path)


# --------------------------------------------------------------------------- #
# ONNX (optional)
# --------------------------------------------------------------------------- #
def export_onnx(model, out_dir: Path, name: str, n_features: int):
    """Export a single model to ONNX, dispatching on its concrete type.

    XGBoost/LightGBM go through ``onnxmltools`` (their native converters);
    scikit-learn estimators (e.g. RandomForest) go through ``skl2onnx``.
    Raises ``RuntimeError`` with a clear message if ONNX tooling is missing.
    """
    try:
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType
    except ImportError as e:
        raise RuntimeError(
            "ONNX export needs: pip install onnx onnxruntime skl2onnx onnxmltools"
        ) from e

    initial_type = [("input", FloatTensorType([None, n_features]))]

    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    if isinstance(model, XGBClassifier):
        onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_type, target_opset=15)
    elif isinstance(model, LGBMClassifier):
        # zipmap=False emits a clean (N, n_classes) probability tensor instead of
        # a sequence of per-class maps.
        onnx_model = onnxmltools.convert_lightgbm(
            model, initial_types=initial_type, target_opset=15, zipmap=False
        )
    else:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType as SkFloatTensorType

        onnx_model = convert_sklearn(
            model, initial_types=[("input", SkFloatTensorType([None, n_features]))], target_opset=15
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.onnx"
    with open(path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    return str(path)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def export_all(trained: dict, metrics_df=None, class_names=None, feature_names=None,
               out_dir=None, onnx: bool = False) -> dict:
    """Export the production artifacts for every model in ``trained``.

    Returns a dict mapping artifact kind -> path.
    """
    out_dir = Path(out_dir or config.OUTPUT_DIR) / EXPORT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = list(class_names or [])
    feature_names = list(feature_names or [])
    n_features = len(feature_names)
    model_entries: list[dict] = []
    artifacts: dict = {}

    for name, model in trained.items():
        model_dir = out_dir / name.replace(" ", "_").replace("-", "_").lower()
        model_dir.mkdir(parents=True, exist_ok=True)

        if name == "XGBoost":
            path = export_xgboost_native(model, model_dir)
            fmt = "xgboost-json"
        elif name == "LGBM":
            path = export_lightgbm_native(model, model_dir)
            fmt = "lightgbm-text"
        elif name == "CatBoost":
            path = export_catboost_native(model, model_dir)
            fmt = "catboost-cbm"
        elif name == "Random Forest":
            path = export_random_forest_native(model, model_dir)
            fmt = "joblib"
        elif "Hybrid" in name:
            # Decompose the custom ensemble into its native base learners + spec.
            base_specs = []
            if hasattr(model, "xgb"):
                p = export_xgboost_native(model.xgb, model_dir)
                base_specs.append({"name": "xgboost", "file": Path(p).name, "format": "xgboost-json"})
            if hasattr(model, "lgbm"):
                p = export_lightgbm_native(model.lgbm, model_dir)
                base_specs.append({"name": "lightgbm", "file": Path(p).name, "format": "lightgbm-text"})
            spec_path = export_ensemble_spec(model_dir, base_specs, class_names, feature_names)
            path = spec_path
            fmt = "ensemble-spec"
            for b in base_specs:
                model_entries.append({
                    "name": f"{name}:{b['name']}",
                    "file": str(model_dir / b["file"]),
                    "format": b["format"],
                })
        else:
            continue

        model_entries.append({"name": name, "file": path, "format": fmt})
        artifacts[name] = path

        if onnx and name in ("XGBoost", "LGBM"):
            try:
                artifacts[f"{name} (onnx)"] = export_onnx(model, model_dir, name.lower(), n_features)
                model_entries.append({
                    "name": f"{name} (onnx)", "file": artifacts[f"{name} (onnx)"], "format": "onnx",
                })
            except Exception as e:  # ONNX is best-effort; never block native export
                artifacts[f"{name} (onnx)"] = f"skipped ({type(e).__name__}): {e}"

    # Metadata (use the hybrid's metrics row if present).
    metrics = {}
    if metrics_df is not None:
        for row_name in ("Hybrid XGB-LGBM (calibrated)", "Hybrid XGB-LGBM"):
            if row_name in metrics_df.index:
                metrics = metrics_df.loc[row_name].to_dict()
                break
        if not metrics:
            metrics = metrics_df.iloc[-1].to_dict()

    metadata_path = build_metadata(out_dir, model_entries, metrics, class_names, feature_names)
    artifacts["metadata"] = metadata_path

    return artifacts
