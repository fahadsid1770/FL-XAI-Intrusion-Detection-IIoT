"""Dataset ingestion for the Edge-IIoTset ML dataset."""
from __future__ import annotations

import pandas as pd

from . import config


def load_raw_dataset(path=None) -> pd.DataFrame:
    """Load the raw Edge-IIoTset ML CSV into a pandas DataFrame.

    Parameters
    ----------
    path : path-like, optional
        Override the default dataset location from ``config``.

    Returns
    -------
    pandas.DataFrame
        The raw dataset (63 columns).
    """
    path = path or config.RAW_DATASET_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place the Edge-IIoTset "
            "'ML-EdgeIIoT-dataset.csv' under data/ first."
        )
    df = pd.read_csv(path, low_memory=False)
    return df


def summarize_classes(df: pd.DataFrame, target: str = config.ATTACK_TYPE_COLUMN) -> pd.DataFrame:
    """Return per-class sample counts and proportions for the target column."""
    counts = df[target].value_counts()
    summary = pd.DataFrame(
        {
            "count": counts,
            "proportion": counts / counts.sum(),
        }
    )
    return summary
