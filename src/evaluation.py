"""Evaluation: classification metrics, confusion matrices, efficiency benchmarks."""
from __future__ import annotations

import time
import tracemalloc

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from . import config


def compute_metrics(y_true, y_pred, average: str = "weighted") -> dict:
    """Return the paper's classification metrics (Eq. 7-10)."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }


def evaluate_model(model, X, y, average: str = "weighted") -> dict:
    """Predict and score a single fitted model."""
    y_pred = model.predict(X)
    return compute_metrics(y, y_pred, average=average)


def build_metrics_table(models: dict, X, y, average: str = "weighted") -> pd.DataFrame:
    """Score a {name: fitted_model} mapping and return a comparison table.

    Rows are models, columns are the metrics from ``config.METRIC_NAMES``.
    """
    rows = {}
    for name, model in models.items():
        rows[name] = evaluate_model(model, X, y, average=average)
    df = pd.DataFrame(rows).T
    return df[list(config.METRIC_NAMES)]


def build_confusion_matrix(y_true, y_pred, labels=None) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=labels)


def time_inference(model, X) -> tuple[float, float]:
    """Time model inference on X.

    Returns (total_seconds, seconds_per_sample).
    """
    start = time.perf_counter()
    model.predict(X)
    total = time.perf_counter() - start
    return total, total / len(X)


def memory_footprint_kb(model, X) -> float:
    """Peak memory (KB) allocated during a single prediction pass via tracemalloc."""
    tracemalloc.start()
    try:
        model.predict(X)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak / 1024.0


def measure_efficiency(model, X_test) -> dict:
    """Measure inference time per sample, memory, and total inference time."""
    model.predict(X_test[: min(64, len(X_test))])  # warm-up to avoid one-off cost
    total_inference_s, seconds_per_sample = time_inference(model, X_test)
    memory_kb = memory_footprint_kb(model, X_test)
    return {
        "inference_time_per_sample_ms": seconds_per_sample * 1000.0,
        "memory_footprint_kb": memory_kb,
        "total_inference_time_s": total_inference_s,
    }
