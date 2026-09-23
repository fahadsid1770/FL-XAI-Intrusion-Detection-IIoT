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

## Production export

`src/export.py` emits portable, language/runtime-agnostic artifacts to
`outputs/export/` (rather than pickle-only `.joblib`):

| Artifact | Format |
|---|---|
| `xgboost.json` | XGBoost native JSON |
| `lightgbm.txt` | LightGBM native text |
| `catboost.cbm` | CatBoost native |
| `*.onnx` | ONNX (optional) |
| `ensemble_spec.json` | soft-voting combination spec (weights, class/feature order) |
| `metadata.json` | provenance: SHA-256 hashes, metrics, feature/class names, library versions |

```python
from src import export
artifacts = export.export_all(trained, metrics_df=metrics_df,
                              class_names=pre.class_names,
                              feature_names=pre.feature_names, onnx=True)
```

ONNX export is optional and degrades gracefully. Install its dependencies with
`pip install -r requirements-export.txt`.

## Key configuration

All seeds use `RANDOM_STATE = 42`. The optimized hyperparameters (paper
Table I) live in `src/config.py`. The pipeline is designed to be extended;
each stage is an isolated module so future work (new models, adaptive learning,
active defense) can plug in without touching the orchestration.

## Dataset notes & decisions

- **MITM** and **Fingerprinting** are dropped as extremely underrepresented
  classes (the paper names MITM explicitly).
- **Ransomware** is recoded to a generalized **Zero_Day** label.
- Identifier / free-text columns (IPs, timestamps, raw payloads, URLs) are
  dropped; low-cardinality categorical features are label-encoded.
- SMOTE (`k=5`) oversamples minority classes to the majority-class count.
- Train/test is a stratified 80/20 split **after** balancing, per the paper.
