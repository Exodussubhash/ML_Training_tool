import mlflow
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def log_to_mlflow(config, models, results, cv_results, experiment_name):
    mlflow.set_experiment(experiment_name)
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            # Params: the model's hyperparameters from config
            model_cfg = next(m for m in config["training"]["models"] if m["name"] == name)
            mlflow.log_params(model_cfg.get("hyperparameters", {}))
            mlflow.log_param("model_type", model_cfg["type"])

            # CV score (if available)
            if name in cv_results:
                mlflow.log_metric("cv_mean", cv_results[name]["f1_mean"])
                mlflow.log_metric("cv_std", cv_results[name]["f1_std"])

            # Test metrics
            r = results[name]
            mlflow.log_metric("accuracy", r["accuracy"])
            mlflow.log_metric("precision", r["precision"])
            mlflow.log_metric("recall", r["recall"])
            mlflow.log_metric("f1_score", r["f1_score"])

            # The model itself
            mlflow.sklearn.log_model(model, name)
    logger.info(f"Logged {len(models)} runs to MLflow experiment '{experiment_name}'")