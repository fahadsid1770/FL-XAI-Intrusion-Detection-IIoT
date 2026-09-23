"""Explainable AI via SHAP TreeExplainer (paper Section III / Phase 7)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from . import config


def compute_shap_values(model, X):
    """Compute raw SHAP values with TreeExplainer.

    For a multi-class tree model this returns a list of arrays (one per class);
    for binary models a single array.
    """
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    return values, explainer


def aggregate_feature_importance(shap_values, feature_names) -> pd.DataFrame:
    """Collapse per-class SHAP values into a single mean-|SHAP| ranking.

    Parameters
    ----------
    shap_values : ndarray or list[ndarray]
        SHAP values of shape (n_classes, n_samples, n_features) or
        (n_samples, n_features).
    feature_names : list[str]

    Returns
    -------
    pandas.DataFrame
        Columns ``feature`` and ``mean_abs_shap``, sorted descending.
    """
    # Normalize to (n_classes, n_samples, n_features), tolerating the three
    # common shap return formats: list[(N,F)], (N,F,C) 3-d array, (N,F) 2-d.
    if isinstance(shap_values, list):
        values = np.stack([np.asarray(v) for v in shap_values], axis=0)  # (C, N, F)
    else:
        values = np.asarray(shap_values)
        if values.ndim == 3:
            values = values.transpose(2, 0, 1)           # (N, F, C) -> (C, N, F)
        elif values.ndim == 2:
            values = values[np.newaxis, :, :]            # (N, F) -> (1, N, F)

    importance = np.abs(values).mean(axis=(0, 1))        # mean over classes & samples
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": importance})
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def shap_values_predicted_class(model, shap_values, X) -> np.ndarray:
    """Reduce multi-class SHAP values to a single (n_samples, n_features) matrix.

    For each sample the SHAP row of the *predicted* class is selected, which is
    what a single beeswarm/summary plot needs.
    """
    values = np.asarray(shap_values)
    if values.ndim == 3:
        values = values.transpose(2, 0, 1)                 # (C, N, F)
        pred = model.predict(X)
        return np.stack([values[c][i] for i, c in enumerate(pred)])
    return values


def shap_sample_indices(y, n_per_class: int = 200, random_state: int = config.RANDOM_STATE) -> np.ndarray:
    """Stratified sample of indices for tractable SHAP computation."""
    rng = np.random.default_rng(random_state)
    indices = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        n = min(n_per_class, len(cls_idx))
        indices.append(rng.choice(cls_idx, size=n, replace=False))
    return np.concatenate(indices)
