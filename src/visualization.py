"""Visualization helpers: distributions, confusion matrices, SHAP, federated."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Calibrated colour palette (viridis-based, consistent across figures).
CLASS_COLORS = plt.cm.viridis

sns.set_theme(style="whitegrid", context="notebook")


def _finalize(ax, title: str, xlabel: str | None = None, ylabel: str | None = None):
    ax.set_title(title, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    return ax


def plot_class_distribution(counts: dict, title: str, ax=None, kind: str = "bar"):
    """Plot class sample counts (before/after SMOTE)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    names = list(counts.keys())
    values = list(counts.values())
    colors = CLASS_COLORS(np.linspace(0, 1, len(names)))
    ax.bar(names, values, color=colors)
    _finalize(ax, title, "Class", "Sample count")
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=7)
    return ax


def plot_confusion_matrix(cm, class_names, title: str, ax=None):
    """Plot a normalized multi-class confusion matrix as a heatmap."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="viridis",
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                cbar_kws={"label": "Normalized"})
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.tick_params(axis="x", rotation=90)
    ax.tick_params(axis="y", rotation=0)
    return ax


def plot_shap_bar(importance_df, top_k: int, title: str, ax=None):
    """Horizontal bar chart of the top-k mean-|SHAP| feature importances."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    top = importance_df.head(top_k).iloc[::-1]  # reverse so largest on top
    colors = CLASS_COLORS(np.linspace(0.2, 0.9, len(top)))
    ax.barh(top["feature"], top["mean_abs_shap"], color=colors)
    _finalize(ax, title, "Mean |SHAP value|", "Feature")
    ax.tick_params(axis="y", rotation=0)
    return ax


def plot_federated_importance(importances, feature_names, top_k: int, ax=None):
    """Heatmap of per-client feature importances to show cross-node consistency."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    means = importances.mean(axis=0)
    top_idx = np.argsort(means)[::-1][:top_k]
    data = importances[:, top_idx]
    names = [feature_names[i] for i in top_idx]
    sns.heatmap(data, xticklabels=names, yticklabels=[f"Client {i+1}" for i in range(len(data))],
                cmap="viridis", ax=ax, cbar_kws={"label": "Importance"})
    ax.set_title("Federated feature importance (per client)", fontweight="bold")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Client node")
    ax.tick_params(axis="x", rotation=90)
    return ax
