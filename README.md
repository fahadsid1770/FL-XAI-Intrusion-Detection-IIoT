# FL-XAI Intrusion Detection for IIoT

A faithful, modular reproduction of *"A Federated and Explainable Machine
Learning Framework for Robust Intrusion Detection and Network Security
Enhancement in Industrial Internet of Things (IIoT) Environments"*
(ICECTE 2026).

The pipeline reflects the paper's full methodology on the **Edge-IIoTset**
dataset: preprocessing and class abstraction, SMOTE balancing, Random-Forest
feature ranking, Optuna (TPE) hyperparameter optimization, a hybrid XGBoost +
LightGBM soft-voting ensemble with isotonic calibration, a federated-learning
simulation over heterogeneous non-IID clients, and SHAP-based explainability.

## Layout

```
FL-XAI-Intrusion-Detection-IIoT/
├── data/                          # Edge-IIoTset ML CSV (download separately)
├── src/                           # modular, extensible pipeline
│   ├── config.py                  # all seeds, paths, hyperparameters
│   ├── data_loader.py             # dataset ingestion
│   ├── preprocessing.py           # cleaning, class abstraction, scaling
│   ├── balancing.py               # SMOTE
│   ├── feature_selection.py       # Random-Forest importance
│   ├── optimization.py            # Optuna TPE search
│   ├── models.py                  # base learners, hybrid, calibration
│   ├── federated.py               # FL simulation (non-IID + FedAvg)
│   ├── explainability.py          # SHAP TreeExplainer
│   ├── evaluation.py              # metrics, efficiency benchmarks
│   ├── visualization.py           # plots
│   ├── persistence.py             # model save/load (joblib)
│   ├── export.py                  # production export (native/ONNX/manifest)
│   └── pipeline.py                # end-to-end orchestration
├── FL-XAI-intrusion-detection.ipynb  # notebook entrypoint
├── outputs/                       # generated tables/figures/models/export
├── requirements.txt               # core dependencies
└── requirements-export.txt        # optional ONNX export dependencies
```

## Setup

A Python 3.12 environment is required.

```bash
# 1. create the environment and install dependencies (uv recommended)
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# macOS only: the xgboost/lightgbm wheels need the OpenMP runtime.
# If it is not already installed, point them at a bundled libomp:
mkdir -p /opt/homebrew/opt/libomp/lib
ln -sf "$(python -c 'import sklearn, os; print(os.path.join(os.path.dirname(sklearn.__file__), ".dylibs/libomp.dylib"))')" \
  /opt/homebrew/opt/libomp/lib/libomp.dylib
```

## Data

The dataset is `ML-EdgeIIoT-dataset.csv` (63 columns, 157,800 rows), the ML
subset of the Edge-IIoTset *"Selected dataset for ML and DL"*. Place it at
`data/ML-EdgeIIoT-dataset.csv` or point `config.RAW_DATASET_PATH` elsewhere.

A direct copy is available on Hugging Face:

```bash
curl -L -o data/ML-EdgeIIoT-dataset.csv \
  "https://huggingface.co/datasets/Sunayanajagadesh/ML_EdgeIIoT_dataset/resolve/main/ML-EdgeIIoT-dataset.csv"
```

## Run

Open `FL-XAI-intrusion-detection.ipynb` and run the cells top to bottom, or
drive the pipeline directly:

```python
from src.pipeline import Pipeline
result = Pipeline().run()
print(result.metrics_df)
```

Trained models are serialized to `outputs/models/*.joblib` on every run. Reload
them without retraining:

```python
from src import persistence
trained, calibrated = persistence.load_all_models()          # everything at once
xgb = persistence.load_model("XGBoost")                     # or a single model
```

## Results

The hybrid XGB-LGBM ensemble is the top
performer across every metric, beating all four baselines.

### Classification metrics (weighted, held-out test set)

| Model | Accuracy | Balanced Acc. | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Random Forest | 0.9298 | 0.9298 | 0.9314 | 0.9298 | 0.9301 |
| CatBoost | 0.9340 | 0.9340 | 0.9379 | 0.9340 | 0.9348 |
| XGBoost | 0.9446 | 0.9446 | 0.9484 | 0.9446 | 0.9453 |
| LGBM | 0.9452 | 0.9452 | 0.9490 | 0.9452 | 0.9460 |
| Hybrid XGB-LGBM | 0.9459 | 0.9459 | 0.9498 | 0.9459 | 0.9466 |
| **Hybrid (calibrated)** | **0.9461** | **0.9461** | **0.9502** | **0.9461** | **0.9469** |

Isotonic calibration (v=5) adds a small, consistent gain over the raw ensemble.
Values sit slightly above the paper's reported targets (e.g. hybrid accuracy
0.9371) because the ML subset is used here rather than the authors' exact data
split — the *relative* ordering and methodology match the paper.

### Computational efficiency

| Model | Inference (ms/sample) | Memory (KB) | Total inference (s) | Training (s) |
|---|---|---|---|---|
| Random Forest | 0.0045 | 73,459 | 0.285 | 6.1 |
| CatBoost | 0.0009 | 494 | 0.059 | 65.2 |
| XGBoost | 0.0063 | 3,708 | 0.398 | 14.2 |
| LGBM | 0.0355 | 7,530 | 2.245 | 10.6 |
| Hybrid XGB-LGBM | 0.0406 | 16,113 | 2.567 | 24.8 |

Memory is measured with `tracemalloc` during inference (the paper's method);
inference latency is per-sample on the full test set. The hybrid stays within
real-time constraints (sub-millisecond per sample).



## Dataset notes & decisions

- **MITM** and **Fingerprinting** are dropped as extremely underrepresented
  classes (the paper names MITM explicitly).
- **Ransomware** is recoded to a generalized **Zero_Day** label.
- Identifier / free-text columns (IPs, timestamps, raw payloads, URLs) are
  dropped; low-cardinality categorical features are label-encoded.
- SMOTE (`k=5`) oversamples minority classes to the majority-class count.
- Train/test is a stratified 80/20 split **after** balancing, per the paper.
