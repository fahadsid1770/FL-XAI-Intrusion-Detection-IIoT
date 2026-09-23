"""End-to-end orchestration of the FL-XAI intrusion-detection pipeline.

Binds the modular stages together and exposes a single ``Pipeline.run()``
entrypoint used by the notebook. Every stage writes its artifacts to
``config.OUTPUT_DIR`` so results are inspectable and extensible.
"""
from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import (
    balancing,
    config,
    data_loader,
    evaluation,
    explainability,
    federated,
    feature_selection,
    models,
    persistence,
    preprocessing,
)


@dataclass
class PipelineResult:
    """Container for all artifacts produced by a pipeline run."""
    # data
    preprocessed: preprocessing.PreprocessingResult
    importance_df: pd.DataFrame
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    class_names: list[str]
    feature_names: list[str]

    # models
    trained_models: dict = field(default_factory=dict)
    calibrated_hybrid: object = None
    model_paths: dict = field(default_factory=dict)
    metrics_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    efficiency_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    # SHAP / FL artifacts
    shap: dict = field(default_factory=dict)
    federated: dict = field(default_factory=dict)


class Pipeline:
    """Reproducible FL-XAI pipeline."""

    def __init__(self, dataset_path=None, top_k: int | None = config.TOP_K_FEATURES,
                 fl_clients: int = config.FL_N_CLIENTS, fl_alpha: float = config.FL_NON_IID_ALPHA):
        self.dataset_path = dataset_path or config.RAW_DATASET_PATH
        self.top_k = top_k
        self.fl_clients = fl_clients
        self.fl_alpha = fl_alpha
        self.result: PipelineResult | None = None

    # -- data ------------------------------------------------------------- #
    def load_and_preprocess(self):
        raw = data_loader.load_raw_dataset(self.dataset_path)
        raw_counts = raw[config.ATTACK_TYPE_COLUMN].value_counts().to_dict()
        preprocessed = preprocessing.preprocess(raw, top_k=self.top_k)
        return raw_counts, preprocessed

    def balance_and_split(self, preprocessed):
        X_bal, y_bal = balancing.apply_smote(preprocessed.X, preprocessed.y)
        importance_df = feature_selection.compute_importances(
            X_bal, y_bal, preprocessed.feature_names
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X_bal, y_bal,
            test_size=config.TEST_SIZE,
            stratify=y_bal,
            random_state=config.RANDOM_STATE,
        )
        return X_bal, y_bal, importance_df, X_train, X_test, y_train, y_test

    # -- training --------------------------------------------------------- #
    def train_models(self, X_train, y_train, calibrated: bool = True):
        """Train baselines, the hybrid ensemble, and (optionally) calibrate it."""
        trained = {}
        train_times = {}

        t0 = time.perf_counter()
        rf = models.build_random_forest().fit(X_train, y_train)
        train_times["Random Forest"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        cat = models.build_catboost().fit(X_train, y_train)
        train_times["CatBoost"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        xgb = models.build_xgboost().fit(X_train, y_train)
        train_times["XGBoost"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        lgbm = models.build_lightgbm().fit(X_train, y_train)
        train_times["LGBM"] = time.perf_counter() - t0

        trained["Random Forest"] = rf
        trained["CatBoost"] = cat
        trained["XGBoost"] = xgb
        trained["LGBM"] = lgbm

        # Hybrid soft-voting ensemble (reuses the trained base learners).
        hybrid = models.SoftVotingEnsemble(xgb=xgb, lgbm=lgbm)
        hybrid.fit(X_train, y_train)
        trained["Hybrid XGB-LGBM"] = hybrid
        train_times["Hybrid XGB-LGBM"] = train_times["XGBoost"] + train_times["LGBM"]

        calibrated_hybrid = None
        if calibrated:
            t0 = time.perf_counter()
            calibrated_hybrid = models.calibrate(
                models.build_hybrid(), X_train, y_train, cv=config.CALIBRATION_CV
            )
            train_times["Hybrid (calibrated)"] = time.perf_counter() - t0

        return trained, calibrated_hybrid, train_times

    # -- evaluation ------------------------------------------------------- #
    def evaluate(self, trained, calibrated_hybrid, X_test, y_test, train_times):
        metrics_df = evaluation.build_metrics_table(trained, X_test, y_test)

        if calibrated_hybrid is not None:
            cal_metrics = evaluation.evaluate_model(calibrated_hybrid, X_test, y_test)
            metrics_df.loc["Hybrid XGB-LGBM (calibrated)"] = cal_metrics

        # Efficiency benchmarks (inference + memory for each model).
        efficiency_rows = {}
        for name, model in trained.items():
            row = evaluation.measure_efficiency(model, X_test)
            row["training_time_s"] = train_times.get(name, np.nan)
            efficiency_rows[name] = row
        efficiency_df = pd.DataFrame(efficiency_rows).T

        return metrics_df, efficiency_df

    # -- SHAP ------------------------------------------------------------- #
    def run_shap(self, trained, X_test, y_test, feature_names, n_per_class: int = 200):
        idx = explainability.shap_sample_indices(y_test, n_per_class=n_per_class)
        X_sample = X_test[idx]
        results = {}
        for name in ("XGBoost", "LGBM"):
            model = trained[name]
            values, _ = explainability.compute_shap_values(model, X_sample)
            importance = explainability.aggregate_feature_importance(values, feature_names)
            results[name] = {"shap_values": values, "importance": importance, "X_sample": X_sample}
        return results

    # -- federated -------------------------------------------------------- #
    def run_federated(self, X_train, y_train, feature_names, learner: str = "hybrid"):
        client_indices = federated.dirichlet_partition(
            y_train, self.fl_clients, self.fl_alpha
        )
        client_models = federated.train_federated_clients(
            X_train, y_train, client_indices, learner=learner
        )
        client_importances = federated.client_feature_importances(
            X_train, y_train, client_indices, feature_names
        )
        return {
            "client_indices": client_indices,
            "client_models": client_models,
            "client_importances": client_importances,
            "shard_sizes": [len(i) for i in client_indices],
        }

    # -- main ------------------------------------------------------------- #
    def run(self, calibrated: bool = True, shap: bool = True, federated: bool = True,
            save_models: bool = True) -> PipelineResult:
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        raw_counts, preprocessed = self.load_and_preprocess()
        X_bal, y_bal, importance_df, X_train, X_test, y_train, y_test = (
            self.balance_and_split(preprocessed)
        )

        trained, calibrated_hybrid, train_times = self.train_models(
            X_train, y_train, calibrated=calibrated
        )
        metrics_df, efficiency_df = self.evaluate(
            trained, calibrated_hybrid, X_test, y_test, train_times
        )

        result = PipelineResult(
            preprocessed=preprocessed,
            importance_df=importance_df,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            class_names=preprocessed.class_names,
            feature_names=preprocessed.feature_names,
            trained_models=trained,
            calibrated_hybrid=calibrated_hybrid,
            metrics_df=metrics_df,
            efficiency_df=efficiency_df,
        )

        if shap:
            result.shap = self.run_shap(
                trained, X_test, y_test, preprocessed.feature_names
            )
        if federated:
            result.federated = self.run_federated(
                X_train, y_train, preprocessed.feature_names
            )

        # Persist key tables.
        metrics_df.to_csv(config.OUTPUT_DIR / "metrics.csv")
        efficiency_df.to_csv(config.OUTPUT_DIR / "efficiency.csv")
        importance_df.to_csv(config.OUTPUT_DIR / "feature_importance.csv", index=False)

        # Persist trained models for reuse without retraining.
        if save_models:
            result.model_paths = persistence.save_all_models(
                trained, calibrated_hybrid, config.MODELS_DIR
            )

        self.result = result
        return result

    def load_models(self):
        """Reload previously persisted models without retraining."""
        return persistence.load_all_models(config.MODELS_DIR)
