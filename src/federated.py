"""Federated-learning simulation (paper Section III / Phase 6).

Simulates N decentralized IIoT edge nodes by partitioning the training data into
heterogeneous non-IID subsets (Dirichlet sampling), training a local model on
each node, and aggregating predictions via soft voting (federated ensemble /
FedAvg-style averaging) without sharing raw data.
"""
from __future__ import annotations

import numpy as np

from . import config
from .models import build_hybrid, build_xgboost, build_lightgbm


def dirichlet_partition(y, n_clients: int, alpha: float, random_state: int = config.RANDOM_STATE) -> list[np.ndarray]:
    """Partition sample indices across clients with a Dirichlet distribution.

    A smaller ``alpha`` yields more heterogeneous (non-IID) client shards.
    """
    rng = np.random.default_rng(random_state)
    n_classes = len(np.unique(y))
    # Per-class, draw a client-assignment proportion from Dirichlet(alpha).
    client_indices = [[] for _ in range(n_clients)]
    for cls in range(n_classes):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        proportions = rng.dirichlet(np.full(n_clients, alpha))
        splits = (proportions * len(cls_idx)).astype(int)
        splits[-1] = len(cls_idx) - splits[:-1].sum()  # fix rounding remainder
        start = 0
        for c in range(n_clients):
            client_indices[c].append(cls_idx[start : start + splits[c]])
            start += splits[c]

    return [np.concatenate(parts) for parts in client_indices]


def train_federated_clients(X, y, client_indices, learner: str = "hybrid") -> list:
    """Train a local model on each client's non-IID shard.

    ``learner`` is one of ``"hybrid"``, ``"xgboost"``, ``"lightgbm"``.
    """
    models = []
    for idx in client_indices:
        if learner == "hybrid":
            model = build_hybrid()
        elif learner == "xgboost":
            model = build_xgboost()
        elif learner == "lightgbm":
            model = build_lightgbm()
        else:
            raise ValueError(f"Unknown learner: {learner}")

        model.fit(X[idx], y[idx])
        models.append(model)
    return models


def federated_predict_proba(models: list, X) -> np.ndarray:
    """Aggregate client predictions by soft-voting (average probabilities)."""
    probas = [m.predict_proba(X) for m in models]
    return np.mean(probas, axis=0)


def federated_predict(models: list, X) -> np.ndarray:
    proba = federated_predict_proba(models, X)
    return np.argmax(proba, axis=1)


def client_feature_importances(X, y, client_indices, feature_names) -> np.ndarray:
    """Return a (n_clients, n_features) matrix of RF importances per client.

    Used to verify that feature-importance rankings stay consistent across
    decentralized nodes (paper Fig. 4).
    """
    from sklearn.ensemble import RandomForestClassifier

    importances = []
    for idx in client_indices:
        rf = RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_STATE, n_jobs=-1)
        rf.fit(X[idx], y[idx])
        importances.append(rf.feature_importances_)
    return np.array(importances)
