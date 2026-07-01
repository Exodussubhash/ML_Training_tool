
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import numpy as np

from src.preprocessing import Preprocessor
from src.utils import load_config
from src.logger import setup_logging
import logging

setup_logging()
logger = logging.getLogger(__name__)

config = load_config("configs/train.yaml")
 
# Registry: add a model type here to make it available in config
MODEL_REGISTRY = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "adaboost": AdaBoostClassifier,
    "decision_tree": DecisionTreeClassifier,
    "knn": KNeighborsClassifier,
    "svc": SVC,
    "naive_bayes": GaussianNB,
}

class Trainer:
    def __init__(self, config, pre_config=None):
        self.target_column = config["target_column"]
        self.model_configs = config["models"]
        self.trained_models = {}
        self.cv_folds = config.get("cv_folds", 0)
        self.pre_config = pre_config
        self.cv_results = {}
 
    def split_features_target(self, df):
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        return X, y
 
    def build_model(self, model_type, hyperparameters):
        if model_type not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model type '{model_type}'. "
                f"Available: {list(MODEL_REGISTRY.keys())}"
            )
        return MODEL_REGISTRY[model_type](**hyperparameters)
    
    def cross_validate_model(self, raw_train_df, model_type, hyperparameters):
        if self.pre_config is None:
            raise ValueError("pre_config is required when cv_folds > 1")

        pre_logger = logging.getLogger("src.preprocessing")
        original_level = pre_logger.level
        pre_logger.setLevel(logging.WARNING)
        try:
            skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            target = self.target_column
            X_idx = raw_train_df.reset_index(drop=True)
            y = X_idx[target]
            scores = []
            for fold, (tr_idx, val_idx) in enumerate(skf.split(X_idx, y)):
                fold_train = X_idx.iloc[tr_idx].copy()
                fold_val = X_idx.iloc[val_idx].copy()
                # Fresh preprocessor per fold; fit on fold-train only.
                pre = Preprocessor(self.pre_config)
                train_proc = pre.fit_transform(fold_train)
                val_proc = pre.transform(fold_val)
                X_tr = train_proc.drop(columns=[target])
                y_tr = train_proc[target]
                X_val = val_proc.drop(columns=[target])
                y_val = val_proc[target]
                model = self.build_model(model_type, hyperparameters)
                model.fit(X_tr, y_tr)
                score = f1_score(y_val, model.predict(X_val), average="weighted")
                scores.append(score)
                logger.info(f"  fold {fold+1}: f1={score:.4f}")
            return float(np.mean(scores)), float(np.std(scores))
        finally:
            pre_logger.setLevel(original_level)

    def train(self, train_df, raw_train_df=None):
        X, y = self.split_features_target(train_df)
        for cfg in self.model_configs:
            name = cfg["name"]
            model_type = cfg["type"]
            hyperparameters = cfg.get("hyperparameters", {})
            if self.cv_folds > 1 and raw_train_df is not None:
                mean, std = self.cross_validate_model(raw_train_df, model_type, hyperparameters)
                self.cv_results[name] = {"f1_mean": mean, "f1_std": std}
                logger.info(f"'{name}' CV f1: {mean:.4f} (+/- {std:.4f})")

            logger.info(f"Training '{name}' ({model_type}) with {hyperparameters}")
            model = self.build_model(model_type, hyperparameters)
            model.fit(X, y)
            self.trained_models[name] = model
            logger.info(f"Finished training '{name}'")
        return self.trained_models
 
