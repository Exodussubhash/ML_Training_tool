"""
Entry point for the ML training pipeline.

Flow: load -> split -> preprocess -> train -> evaluate -> save artifacts.
The config path is passed as a CLI argument, so the same tool runs any
dataset / model setup defined in YAML.

Usage:
    python main.py --config configs/train.yaml
"""

import argparse
import logging

from sklearn.model_selection import train_test_split

from src.utils import load_config
from src.loader import load_data
from src.preprocessing import Preprocessor
from src.trainer import Trainer
from src.evaluator import Evaluator
from artifacts.create_artifacts import ArtifactStore
from src.logger import setup_logging
from src.mlflow_log import log_to_mlflow

setup_logging()
logger = logging.getLogger(__name__)



def parse_args():
    parser = argparse.ArgumentParser(description="ML training pipeline")
    parser.add_argument(
        "--config",
        default="configs/train.yaml",
        help="Path to the YAML config file.",
    )
    return parser.parse_args()


def split_data(df, target, split_cfg):
    """Split into train and test, optionally stratified on the target."""
    stratify = df[target] if split_cfg.get("stratify") else None
    train_df, test_df = train_test_split(
        df,
        test_size=split_cfg["test_size"],
        random_state=split_cfg["random_state"],
        stratify=stratify,
    )
    logger.info(f"Split data: {len(train_df)} train, {len(test_df)} test")
    return train_df, test_df


def run_pipeline(config):
    pre_config = config["preprocessing"]
    split_cfg = config["split"]
    train_config = config["training"]
    target = pre_config["target_column"]
    train_config["target_column"] = target

    # Load
    df = load_data(config["data_path"], target)

    # Split (before preprocessing, so the preprocessor only fits on train)
    train_df, test_df = split_data(df, target, split_cfg)

    # Preprocess
    preprocessor = Preprocessor(pre_config)
    train_processed = preprocessor.fit_transform(train_df)
    test_processed = preprocessor.transform(test_df)

    # Train
    trainer = Trainer(train_config, pre_config)
    models = trainer.train(train_processed, raw_train_df=train_df)

    # Evaluate
    evaluator = Evaluator(models, test_processed)
    results = evaluator.evaluate()
    for name, cv_result in trainer.cv_results.items():
        if name in results:
            results[name]["cross_validation"] = cv_result

    # Log to MLflow
    if config["use_mlflow"]:
        experiment_name = config.get("mlflow_experiment_name", "default")
        log_to_mlflow(config, models, results, trainer.cv_results, experiment_name)

    # Save versioned artifacts
    store = ArtifactStore(config.get("artifacts_dir", train_config.get("artifacts_dir", "artifacts")))
    run_dir = store.save(
        models, results, config,
        feature_columns=list(train_processed.columns),
    )
    logger.info(f"Pipeline complete. Artifacts saved to {run_dir}")
    return results


def main():
    args = parse_args()
    config = load_config(args.config)
    run_pipeline(config)


if __name__ == "__main__":
    main()
