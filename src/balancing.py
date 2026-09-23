"""Class balancing via SMOTE (Synthetic Minority Over-sampling Technique)."""
from __future__ import annotations

from imblearn.over_sampling import SMOTE

from . import config


def apply_smote(X, y, k_neighbors: int = config.SMOTE_K_NEIGHBORS, random_state: int = config.RANDOM_STATE):
    """Oversample minority classes until approximate class parity.

    Uses k-nearest-neighbour interpolation (paper Eq. 2-3, k=5) so synthetic
    samples lie within the minority-class feature space rather than duplicating
    existing rows.
    """
    smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled


def class_counts(y) -> dict:
    """Return class-index -> sample-count mapping."""
    import numpy as np

    values, counts = np.unique(y, return_counts=True)
    return dict(zip(values.tolist(), counts.tolist()))
