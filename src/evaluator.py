from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report, confusion_matrix
from src.utils import load_config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = load_config("configs/train.yaml")
models_name = config["training"]["models"]
target_column = config["preprocessing"]["target_column"]

class Evaluator:
    def __init__(self, models, test_df):
        self.models = models
        self.test_df = test_df
    
    def split_features_target(self, df):
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return X, y

    def evaluate(self):
        X_test, y_test = self.split_features_target(self.test_df)
        n_classes = y_test.nunique()
        average = "binary" if n_classes == 2 else "weighted"
        logger.info(f"Detected {n_classes} classes; using average='{average}'")

        results = {}
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            results[name] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average=average, zero_division=0),
                "recall": recall_score(y_test, y_pred, average=average, zero_division=0),
                "f1_score": f1_score(y_test, y_pred, average=average, zero_division=0),
                "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            }
            logger.info(
                f"Evaluated '{name}': accuracy={results[name]['accuracy']:.4f}, "
                f"precision={results[name]['precision']:.4f}, "
                f"recall={results[name]['recall']:.4f}, "
                f"f1_score={results[name]['f1_score']:.4f}"
            )
        return results