"""Central configuration for the FL-XAI intrusion-detection pipeline.

Every constant that grounds the reproduction in the paper lives here, so the
notebook and scripts stay thin and the pipeline stays reproducible.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
RAW_DATASET_PATH = DATA_DIR / "ML-EdgeIIoT-dataset.csv"

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42

# --------------------------------------------------------------------------- #
# Dataset schema (Edge-IIoTset "Selected dataset for ML and DL")
# --------------------------------------------------------------------------- #
# The two target columns present in the raw CSV.
LABEL_COLUMN = "Attack_label"          # binary (Normal vs Attack)
ATTACK_TYPE_COLUMN = "Attack_type"     # multi-class label (15 classes)

# Classes dropped during cleaning: extremely underrepresented attack classes.
# MITM is named explicitly in the paper; Fingerprinting is even sparser and is
# dropped under the same "exclude extremely sparse classes" rule.
DROPPED_CLASSES = ["MITM", "Fingerprinting"]

# Specialized attack categories recoded into a generalized zero-day label.
ZERO_DAY_SOURCE_CLASSES = ["Ransomware"]
ZERO_DAY_LABEL = "Zero_Day"

# Identifier / timestamp columns that carry no generalisable signal and are
# dropped before modelling (IPs, raw payloads, high-cardinality free text).
DROP_COLUMNS = [
    "frame.time",
    "ip.src_host",
    "ip.dst_host",
    "arp.dst.proto_ipv4",
    "arp.src.proto_ipv4",
    "http.file_data",
    "http.request.uri.query",
    "http.request.full_uri",
    "http.referer",
    "tcp.options",
    "tcp.payload",
    "tcp.srcport",  # corrupted: hostnames mixed with numeric ports (typographical errors)
    "dns.qry.name",
    "mqtt.msg",
    "mqtt.topic",
    "mqtt.msg_decoded_as",
]

# --------------------------------------------------------------------------- #
# Data balancing & splitting
# --------------------------------------------------------------------------- #
SMOTE_K_NEIGHBORS = 5
TEST_SIZE = 0.20
STRATIFY = True

# --------------------------------------------------------------------------- #
# Feature selection
# --------------------------------------------------------------------------- #
# Number of top features kept after Random-Forest importance ranking. None
# keeps every engineered feature (the paper reports SHAP on the full set).
TOP_K_FEATURES: int | None = None

# --------------------------------------------------------------------------- #
# Hyperparameters (paper Table I -- the Optuna TPE search targets)
# --------------------------------------------------------------------------- #
XGB_PARAMS = {
    "n_estimators": 215,
    "max_depth": 15,
    "learning_rate": 0.101,
    "subsample": 0.884,
    "colsample_bytree": 0.809,
    "gamma": 0.187,
    "min_child_weight": 3,
    "tree_method": "hist",
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": 0,
}

LGBM_PARAMS = {
    "n_estimators": 291,
    "max_depth": -1,
    "learning_rate": 0.077,
    "subsample": 0.622,
    "colsample_bytree": 0.898,
    "num_leaves": 74,
    "min_child_samples": 24,
    "reg_alpha": 0.027,
    "reg_lambda": 0.013,
    "objective": "multiclass",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": -1,
}

# Optuna search space bounds (used by the TPE sampler).
XGB_SEARCH_SPACE = {
    "n_estimators": (100, 400),
    "max_depth": (3, 20),
    "learning_rate": (0.01, 0.3),
    "subsample": (0.5, 1.0),
    "colsample_bytree": (0.5, 1.0),
    "gamma": (0.0, 1.0),
    "min_child_weight": (1, 10),
}

LGBM_SEARCH_SPACE = {
    "n_estimators": (100, 500),
    "max_depth": (-1, 30),
    "learning_rate": (0.01, 0.3),
    "subsample": (0.5, 1.0),
    "colsample_bytree": (0.5, 1.0),
    "num_leaves": (16, 256),
    "min_child_samples": (5, 100),
    "reg_alpha": (0.0, 1.0),
    "reg_lambda": (0.0, 1.0),
}

OPTUNA_N_TRIALS = 30
OPTUNA_TIMEOUT_SECONDS: int | None = None

# --------------------------------------------------------------------------- #
# Ensemble & calibration
# --------------------------------------------------------------------------- #
# Soft-voting ensemble of base learners.
BASE_LEARNERS = ("xgboost", "lightgbm")
CALIBRATION_METHOD = "isotonic"
CALIBRATION_CV = 5

# --------------------------------------------------------------------------- #
# Federated learning simulation
# --------------------------------------------------------------------------- #
FL_N_CLIENTS = 4
FL_NON_IID_ALPHA = 0.5  # Dirichlet concentration for non-IID partitioning
FL_ROUNDS = 1            # single round of local training + aggregation

# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
METRIC_NAMES = ["accuracy", "balanced_accuracy", "precision", "recall", "f1"]

# Benchmark targets from paper Table II (for reporting / parity checks).
BENCHMARK_TARGETS = {
    "accuracy": 0.9371,
    "balanced_accuracy": 0.9372,
    "f1": 0.9374,
    "precision": 0.9409,
    "recall": 0.9371,
}

# SHAP visualisation settings.
SHAP_TOP_FEATURES = 20
