"""Hyperparameter optimization with Optuna's TPE sampler (paper Section III.G.1)."""
from __future__ import annotations

import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import f1_score

from . import config

# Optuna's median pruner stops unpromising trials early to keep the search tractable.
_PRUNE_EARLY = False


def _objective_xgboost(trial, X, y):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", *config.XGB_SEARCH_SPACE["n_estimators"]),
        "max_depth": trial.suggest_int("max_depth", *config.XGB_SEARCH_SPACE["max_depth"]),
        "learning_rate": trial.suggest_float("learning_rate", *config.XGB_SEARCH_SPACE["learning_rate"], log=True),
        "subsample": trial.suggest_float("subsample", *config.XGB_SEARCH_SPACE["subsample"]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", *config.XGB_SEARCH_SPACE["colsample_bytree"]),
        "gamma": trial.suggest_float("gamma", *config.XGB_SEARCH_SPACE["gamma"]),
        "min_child_weight": trial.suggest_int("min_child_weight", *config.XGB_SEARCH_SPACE["min_child_weight"]),
        "tree_method": "hist",
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "random_state": config.RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": 0,
    }
    from xgboost import XGBClassifier

    model = XGBClassifier(**params)
    model.fit(X, y)
    pred = model.predict(X)
    return f1_score(y, pred, average="weighted")


def _objective_lightgbm(trial, X, y):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", *config.LGBM_SEARCH_SPACE["n_estimators"]),
        "max_depth": trial.suggest_int("max_depth", *config.LGBM_SEARCH_SPACE["max_depth"]),
        "learning_rate": trial.suggest_float("learning_rate", *config.LGBM_SEARCH_SPACE["learning_rate"], log=True),
        "subsample": trial.suggest_float("subsample", *config.LGBM_SEARCH_SPACE["subsample"]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", *config.LGBM_SEARCH_SPACE["colsample_bytree"]),
        "num_leaves": trial.suggest_int("num_leaves", *config.LGBM_SEARCH_SPACE["num_leaves"]),
        "min_child_samples": trial.suggest_int("min_child_samples", *config.LGBM_SEARCH_SPACE["min_child_samples"]),
        "reg_alpha": trial.suggest_float("reg_alpha", *config.LGBM_SEARCH_SPACE["reg_alpha"]),
        "reg_lambda": trial.suggest_float("reg_lambda", *config.LGBM_SEARCH_SPACE["reg_lambda"]),
        "objective": "multiclass",
        "random_state": config.RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
    }
    from lightgbm import LGBMClassifier

    model = LGBMClassifier(**params)
    model.fit(X, y)
    pred = model.predict(X)
    return f1_score(y, pred, average="weighted")


def run_study(model_name: str, X, y, n_trials: int = config.OPTUNA_N_TRIALS, timeout: int | None = None):
    """Run an Optuna TPE study for one base learner, maximizing weighted F1."""
    objective = {
        "xgboost": _objective_xgboost,
        "lightgbm": _objective_lightgbm,
    }[model_name]

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=config.RANDOM_STATE),
    )
    study.optimize(
        lambda trial: objective(trial, X, y),
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=1,
        show_progress_bar=False,
    )
    return study
