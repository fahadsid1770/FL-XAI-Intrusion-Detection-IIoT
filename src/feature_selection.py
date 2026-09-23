"""Random-Forest feature-importance ranking (paper Section III.D)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from . import config


def compute_importances(X, y, feature_names, random_state: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Fit a preliminary RandomForest and return Gini-importance rankings."""
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X, y)
    importances = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": rf.feature_importances_,
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)
    importances.index = importances.index + 1  # rank
    return importances


def select_top_features(importance_df: pd.DataFrame, top_k: int) -> list[str]:
    """Return the names of the top-k features by importance."""
    return importance_df.head(top_k)["feature"].tolist()


def select_top_indices(feature_names: list[str], importance_df: pd.DataFrame, top_k: int) -> np.ndarray:
    """Return column indices of the top-k features, in original feature order."""
    top_names = set(select_top_features(importance_df, top_k))
    return np.array([i for i, name in enumerate(feature_names) if name in top_names])
