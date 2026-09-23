"""Model definitions: base learners, hybrid soft-voting ensemble, calibration.

The hybrid XGB-LGBM model averages per-class probabilities across the two base
learners (paper Eq. 5) and predicts via argmax; its output confidence is then
calibrated with isotonic regression (paper Eq. 6, v=5).
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from . import config


def _is_fitted(estimator) -> bool:
    try:
        check_is_fitted(estimator)
        return True
    except (NotFittedError, TypeError):
        return False


def build_xgboost(params: dict | None = None) -> XGBClassifier:
    p = dict(config.XGB_PARAMS)
    if params:
        p.update(params)
    return XGBClassifier(**p)


def build_lightgbm(params: dict | None = None) -> LGBMClassifier:
    p = dict(config.LGBM_PARAMS)
    if params:
        p.update(params)
    return LGBMClassifier(**p)


def build_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )


def build_catboost():
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        loss_function="MultiClass",
        random_seed=config.RANDOM_STATE,
        thread_count=-1,
        verbose=0,
        allow_writing_files=False,
    )


class SoftVotingEnsemble(BaseEstimator, ClassifierMixin):
    """Soft-voting ensemble averaging predicted probabilities (paper Eq. 5).

    Exposes ``xgb`` and ``lgbm`` as constructor parameters so the estimator is
    cloneable by ``CalibratedClassifierCV`` (which refits it per fold).
    """

    def __init__(self, xgb: XGBClassifier, lgbm: LGBMClassifier):
        self.xgb = xgb
        self.lgbm = lgbm

    def fit(self, X, y):
        # Reuse pre-fitted base learners when available (avoids a second,
        # wasteful training pass); a freshly cloned estimator still fits them.
        if not _is_fitted(self.xgb):
            self.xgb.fit(X, y)
        if not _is_fitted(self.lgbm):
            self.lgbm.fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        p_xgb = self.xgb.predict_proba(X)
        p_lgbm = self.lgbm.predict_proba(X)
        return (p_xgb + p_lgbm) / 2.0

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


def build_hybrid(xgb: XGBClassifier | None = None, lgbm: LGBMClassifier | None = None) -> SoftVotingEnsemble:
    """Build the soft-voting XGB-LGBM hybrid ensemble."""
    xgb = xgb or build_xgboost()
    lgbm = lgbm or build_lightgbm()
    return SoftVotingEnsemble(xgb=xgb, lgbm=lgbm)


def calibrate(model, X, y, method: str = config.CALIBRATION_METHOD, cv: int = config.CALIBRATION_CV):
    """Wrap a model with isotonic probability calibration (paper Eq. 6)."""
    # n_jobs=1: run calibration folds sequentially so the tree learners inside
    # each fold can use all cores without oversubscribing the CPU.
    calibrated = CalibratedClassifierCV(
        estimator=model,
        method=method,
        cv=cv,
        n_jobs=1,
    )
    calibrated.fit(X, y)
    return calibrated
