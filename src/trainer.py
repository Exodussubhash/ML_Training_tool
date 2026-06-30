
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

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
    def __init__(self, config):
        self.target_column = config["target_column"]
        self.model_configs = config["models"]
        self.trained_models = {}
 
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
 
    def train(self, train_df):
        X, y = self.split_features_target(train_df)
        for cfg in self.model_configs:
            name = cfg["name"]
            model_type = cfg["type"]
            hyperparameters = cfg.get("hyperparameters", {})
            logger.info(f"Training '{name}' ({model_type}) with {hyperparameters}")
            model = self.build_model(model_type, hyperparameters)
            model.fit(X, y)
            self.trained_models[name] = model
            logger.info(f"Finished training '{name}'")
        return self.trained_models
 