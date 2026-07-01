# ML Model Training Pipeline

A configurable, config-driven tool that takes a raw tabular dataset, trains one
or more machine learning models, and outputs versioned model artifacts with
metadata. The entire pipeline is controlled through a YAML config, so the same
tool runs on any classification dataset without code changes.

## Overview

The pipeline performs the following stages:

1. **Load** – read a CSV dataset and validate it.
2. **Split** – split into train and test sets (optionally stratified) *before*
   preprocessing, so the preprocessor only ever fits on training data.
3. **Preprocess** – clean, encode, and scale the data based on config.
4. **Train** – train one or more models defined in config.
5. **Evaluate** – score each model on the test set.
6. **Save** – persist each trained model with its metadata as a versioned artifact.

## Project structure

```
.
├── configs/
│   ├── train.yaml          # full training run config
│   └── test.yaml           # lightweight config for quick runs
├── data/                   # input datasets
├── artifacts/              # versioned model outputs (created at runtime)
│   └── create_artifacts.py # artifact saving/loading
├── src/
│   ├── loader.py           # load + validate CSV
│   ├── preprocessing.py    # config-driven preprocessing (fit/transform)
│   ├── trainer.py          # model registry + training
│   ├── evaluator.py        # metrics + evaluation
│   ├── utils.py            # config loading + helpers
│   └── logger.py           # logging setup
├── tests/
│   └── test_pipeline.py    # unit tests
├── main.py                 # entry point
└── requirements.txt
```

## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the pipeline with a config:

```bash
python main.py --config configs/train.yaml
```

Run the tests:

```bash
python -m pytest tests/test_pipeline.py -v
```

## Configuration

All behaviour is driven by YAML. Example:

```yaml
data_path: "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

preprocessing:
  target_column: "Churn"
  drop_columns: ["customerID"]
  encode_target: true
  missing_strategy: "median"      # median | mean | drop
  encoding: "onehot"              # onehot | label
  scaling: "standard"             # standard | minmax | normalize | none
  log_transform:
    mode: "auto"                  # auto | manual | off
    skew_threshold: 0.75
    columns: []
  outliers:
    handle: false
    multiplier: 1.5
    columns: []                   # empty = all continuous numeric columns

split:
  test_size: 0.2
  random_state: 42
  stratify: true

training:
  models:
    - name: "logistic_regression"
      type: "logistic_regression"
      hyperparameters: {max_iter: 1000, class_weight: "balanced"}
    - name: "random_forest"
      type: "random_forest"
      hyperparameters: {n_estimators: 200, max_depth: 10, class_weight: "balanced"}

artifacts_dir: "artifacts"
```

## Design decisions

**Config-driven, dataset-agnostic.** Every step (target column, dropped columns,
imputation strategy, encoding, scaling, models, hyperparameters) is set in YAML.
The tool was verified on two different datasets — a binary churn dataset and a
4-class customer segmentation dataset — with no code changes, only config.

**Fit/transform separation to prevent leakage.** Preprocessing parameters
(scaler statistics, imputation values, encoder schema) are learned on the
training data only, then applied unchanged to the test data. The train/test
split happens before preprocessing so the test set never influences what is
learned. This avoids data leakage and train/serve skew.

**Schema alignment.** One-hot encoding can produce different columns on train
versus test. The training feature schema is locked during fit, and test data is
re-aligned to it, guaranteeing consistent inputs to the model.

**Model registry.** Models are looked up from a registry by name, so adding a new
model type is a single entry — the training code does not change.

**Binary and multi-class support.** The target is encoded generically with a
label encoder, and evaluation metrics automatically switch averaging strategy
(`binary` for two classes, `weighted` for more), so the tool handles both cases.

**Versioned artifacts.** Each run writes to a timestamped folder containing the
serialized model plus metadata (timestamp, metrics, config, feature columns),
so runs are reproducible and traceable.

**Class imbalance.** Handled via `class_weight="balanced"` set per model in
config, with precision/recall/F1 reported rather than accuracy alone, since
accuracy is misleading on imbalanced data.

## Assumptions

- Input is a tabular CSV with a single target column for a classification task.
- Numeric columns stored as text (e.g. blank strings) are auto-coerced when more
  than 90% of values parse as numbers; otherwise the column is treated as
  categorical.
- Outlier removal applies only to training data and skips low-cardinality
  (flag-like) numeric columns by default.

## What I would improve with more time

- **Experiment tracking** (e.g. MLflow) instead of JSON metadata files.
- **A batch prediction pipeline** that loads a saved artifact and scores new data
  in memory-efficient chunks, plus a monitoring layer for drift detection.
- **Cloud artifact storage** (e.g. GCS) behind the same interface as local
  storage.
- **Config-driven tests** rather than fixed fixtures.
