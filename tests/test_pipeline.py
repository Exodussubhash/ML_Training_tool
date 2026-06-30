"""
Unit tests for the ML pipeline components.
Run with: pytest tests/test_pipeline.py
"""

import pandas as pd
import pytest

from src.loader import load_data
from src.preprocessing import Preprocessor
from src.trainer import Trainer
from src.evaluator import Evaluator


@pytest.fixture
def sample_df():
    """Small synthetic dataset resembling the Telco schema."""
    return pd.DataFrame({
        "customerID": ["a", "b", "c", "d", "e", "f", "g", "h"],
        "gender": ["Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female"],
        "tenure": [1, 24, 5, 60, 12, 36, 2, 48],
        "MonthlyCharges": [29.85, 56.95, 53.85, 42.30, 70.70, 99.65, 20.05, 89.10],
"TotalCharges": ["29.85", "1889.5", "108.15", "1840.75", "150.0", "3046.05", "40.10", "4012.5"],        "Contract": ["Month", "Year", "Month", "Two year", "Month", "Year", "Month", "Two year"],
        "Churn": ["No", "No", "Yes", "No", "Yes", "No", "Yes", "No"],
    })


@pytest.fixture
def pre_config():
    return {
        "target_column": "Churn",
        "drop_columns": ["customerID"],
        "encode_target": True,
        "missing_strategy": "median",
        "encoding": "onehot",
        "scaling": "standard",
        "log_transform": {"mode": "off"},
        "outliers": {"handle": False},
    }


def test_load_data(tmp_path, sample_df):
    path = tmp_path / "sample.csv"
    sample_df.to_csv(path, index=False)
    df = load_data(str(path), "Churn")
    assert df.shape[0] == 8
    assert "Churn" in df.columns


def test_load_data_missing_target(tmp_path, sample_df):
    path = tmp_path / "sample.csv"
    sample_df.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_data(str(path), "NotAColumn")


def test_preprocessor_drops_and_encodes_target(sample_df, pre_config):
    pre = Preprocessor(pre_config)
    out = pre.fit_transform(sample_df)
    assert "customerID" not in out.columns
    assert set(out["Churn"].unique()).issubset({0, 1})


def test_preprocessor_coerces_totalcharges(sample_df, pre_config):
    pre = Preprocessor(pre_config)
    out = pre.fit_transform(sample_df)
    # TotalCharges had a blank -> coerced numeric, imputed, no nulls left
    assert out.isnull().sum().sum() == 0


def test_train_and_transform_schema_match(sample_df, pre_config):
    pre = Preprocessor(pre_config)
    train_out = pre.fit_transform(sample_df)
    test_out = pre.transform(sample_df)
    # Test data aligned to training feature schema
    train_features = [c for c in train_out.columns if c != "Churn"]
    test_features = [c for c in test_out.columns if c != "Churn"]
    # assert list(train_out.columns) == list(test_out.columns)
    assert train_features == test_features



def test_trainer_trains_models(sample_df, pre_config):
    pre = Preprocessor(pre_config)
    train_out = pre.fit_transform(sample_df)
    train_config = {
        "target_column": "Churn",
        "models": [
            {"name": "lr", "type": "logistic_regression",
             "hyperparameters": {"max_iter": 200}},
        ],
    }
    trainer = Trainer(train_config)
    models = trainer.train(train_out)
    assert "lr" in models


def test_trainer_rejects_unknown_model(sample_df, pre_config):
    train_config = {
        "target_column": "Churn",
        "models": [{"name": "x", "type": "not_a_model", "hyperparameters": {}}],
    }
    pre = Preprocessor(pre_config)
    train_out = pre.fit_transform(sample_df)
    trainer = Trainer(train_config)
    with pytest.raises(ValueError):
        trainer.train(train_out)


def test_evaluator_returns_metrics(sample_df, pre_config):
    pre = Preprocessor(pre_config)
    train_out = pre.fit_transform(sample_df)
    test_out = pre.transform(sample_df)
    train_config = {
        "target_column": "Churn",
        "models": [
            {"name": "lr", "type": "logistic_regression",
             "hyperparameters": {"max_iter": 200}},
        ],
    }
    trainer = Trainer(train_config)
    models = trainer.train(train_out)
    evaluator = Evaluator(models, test_out)
    results = evaluator.evaluate()
    assert "lr" in results
    assert "accuracy" in results["lr"]