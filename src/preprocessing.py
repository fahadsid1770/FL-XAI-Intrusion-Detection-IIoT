"""Data cleaning, class abstraction, feature encoding and standardisation.

Implements the preprocessing stages described in the paper (Section III.B/D):
  * drop extremely underrepresented attack classes (MITM, Fingerprinting),
  * recode specialised attacks (Ransomware) into a generalized "Zero Day" label,
  * drop identifier / free-text columns,
  * label-encode low-cardinality categorical features,
  * standard-normalize numeric features (z-score).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from . import config

# Categorical columns above this cardinality are treated as free text and dropped.
CATEGORICAL_MAX_CARDINALITY = 30

# Columns that encode categorical (non-ordinal) protocol metadata and are
# label-encoded rather than standard-scaled.
CATEGORICAL_COLUMNS = [
    "http.request.method",
    "http.request.version",
    "mqtt.protoname",
]


class PreprocessingResult:
    """Container holding the transformed arrays and the fitted transformers."""

    def __init__(self, X, y, feature_names, class_names, label_encoder, scaler, report):
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.class_names = class_names
        self.label_encoder = label_encoder
        self.scaler = scaler
        self.report = report


def _clean_target(df: pd.DataFrame) -> pd.DataFrame:
    """Drop underrepresented classes and recode zero-day classes."""
    df = df[~df[config.ATTACK_TYPE_COLUMN].isin(config.DROPPED_CLASSES)].copy()
    df[config.ATTACK_TYPE_COLUMN] = df[config.ATTACK_TYPE_COLUMN].replace(
        config.ZERO_DAY_SOURCE_CLASSES, config.ZERO_DAY_LABEL
    )
    return df


def _is_numeric_series(series: pd.Series) -> bool:
    """Return True if a series is numeric after coercing empty values."""
    if pd.api.types.is_numeric_dtype(series):
        return True
    coerced = pd.to_numeric(series, errors="coerce")
    # Treat as numeric if at least ~98% of non-null values coerce successfully.
    non_null = series.notna().sum()
    if non_null == 0:
        return False
    return (coerced.notna().sum() / non_null) >= 0.98


def preprocess(df: pd.DataFrame, top_k: int | None = None) -> PreprocessingResult:
    """Run the full preprocessing pipeline and return transformed data.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw dataset from :func:`data_loader.load_raw_dataset`.
    top_k : int, optional
        If given, keep only the top-k most important features (RF selection is
        performed separately; this argument is reserved for downstream use).

    Returns
    -------
    PreprocessingResult
    """
    report: dict[str, object] = {}

    # 1. Class abstraction -------------------------------------------------
    df = _clean_target(df)
    report["classes_after_cleaning"] = df[config.ATTACK_TYPE_COLUMN].value_counts().to_dict()
    report["n_rows_after_cleaning"] = len(df)

    # 2. Drop binary label + identifier / free-text columns -----------------
    columns_to_drop = [
        c for c in [config.LABEL_COLUMN, *config.DROP_COLUMNS] if c in df.columns
    ]
    df = df.drop(columns=columns_to_drop)
    report["dropped_columns"] = columns_to_drop

    # 3. Separate features and target --------------------------------------
    y_series = df[config.ATTACK_TYPE_COLUMN]
    X = df.drop(columns=[config.ATTACK_TYPE_COLUMN])

    # 4. Classify feature columns ------------------------------------------
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    dropped_cols: list[str] = []

    for col in X.columns:
        series = X[col]
        if col in CATEGORICAL_COLUMNS:
            categorical_cols.append(col)
            continue
        if _is_numeric_series(series):
            numeric_cols.append(col)
        elif series.nunique(dropna=True) <= CATEGORICAL_MAX_CARDINALITY:
            categorical_cols.append(col)
        else:
            dropped_cols.append(col)

    report["numeric_features"] = numeric_cols
    report["categorical_features"] = categorical_cols
    report["auto_dropped_high_cardinality"] = dropped_cols

    # 5. Encode categorical features ---------------------------------------
    X_encoded = X.copy()
    for col in categorical_cols:
        X_encoded[col] = LabelEncoder().fit_transform(
            X_encoded[col].fillna("missing").astype(str)
        )

    # 6. Coerce numeric columns and impute missing values ------------------
    for col in numeric_cols:
        X_encoded[col] = pd.to_numeric(X_encoded[col], errors="coerce")

    if X_encoded[numeric_cols].isna().any().any():
        report["numeric_imputation"] = "median"
        X_encoded[numeric_cols] = X_encoded[numeric_cols].fillna(
            X_encoded[numeric_cols].median()
        )

    # 7. Standard normalization --------------------------------------------
    scaler = StandardScaler()
    feature_names = list(X_encoded.columns)
    X_scaled = scaler.fit_transform(X_encoded)

    # 8. Encode target -----------------------------------------------------
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_series)
    class_names = list(label_encoder.classes_)

    report["feature_names"] = feature_names
    report["n_features"] = len(feature_names)

    return PreprocessingResult(
        X=X_scaled,
        y=y,
        feature_names=feature_names,
        class_names=class_names,
        label_encoder=label_encoder,
        scaler=scaler,
        report=report,
    )


def label_distribution_before_after(raw_counts: dict, cleaned_counts: dict) -> pd.DataFrame:
    """Build a small comparison table of class counts before/after cleaning."""
    before = pd.Series(raw_counts, name="before")
    after = pd.Series(cleaned_counts, name="after")
    return pd.concat([before, after], axis=1).fillna(0).astype(int)
