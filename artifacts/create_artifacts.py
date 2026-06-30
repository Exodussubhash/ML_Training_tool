"""
Versioned artifact storage.
Saves each trained model with its metadata (timestamp, metrics, config, feature
columns) so runs are traceable and models can be reloaded for inference.
"""

import os
import json
import logging
import pickle
from datetime import datetime

logger = logging.getLogger(__name__)


class ArtifactStore:
    def __init__(self, artifacts_dir="artifacts"):
        self.artifacts_dir = artifacts_dir
        os.makedirs(self.artifacts_dir, exist_ok=True)

    def save(self, models, metrics, config, feature_columns=None):
        # One timestamped folder per run, so artifacts are versioned and never
        # overwrite previous runs.
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.artifacts_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)

        for name, model in models.items():
            model_path = os.path.join(run_dir, f"{name}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            logger.info(f"Saved model '{name}' to {model_path}")

            metadata = {
                "model_name": name,
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics.get(name, {}),
                "config": config,
                "feature_columns": feature_columns,
            }
            meta_path = os.path.join(run_dir, f"{name}_metadata.json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved metadata for '{name}' to {meta_path}")

        logger.info(f"All artifacts saved under {run_dir}")
        return run_dir

    def load_model(self, run_id, model_name):
        model_path = os.path.join(self.artifacts_dir, run_id, f"{model_name}.pkl")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Loaded model '{model_name}' from {model_path}")
        return model