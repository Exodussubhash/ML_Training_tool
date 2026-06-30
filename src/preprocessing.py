"""
Config-driven preprocessing pipeline for tabular data.
All behaviour is controlled via the config dict (typically from YAML).

Uses fit_transform (on training data) and transform (on test/inference data):
parameters such as scaler statistics, encoder columns and impute values are
learned on training data only, then applied unchanged to test data. This
prevents test data leaking into training and keeps train/test consistent.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, LabelEncoder

logger = logging.getLogger(__name__)


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.target_column = config["target_column"]
        self.drop_columns = config.get("drop_columns", [])
        self.encode_target_flag = config.get("encode_target", True)
        self.missing_strategy = config.get("missing_strategy", "median")
        self.encoding = config.get("encoding", "onehot")
        self.scaling = config.get("scaling", "none")
        self.log_config = config.get("log_transform", {})
        self.outlier_config = config.get("outliers", {})

        # Learned during fit_transform, reused in transform
        self.target_encoder = None
        self.impute_values = {}
        self.log_columns = []
        self.feature_columns = None
        self.scaler = None
        self.scaled_columns = None
        self.label_maps = {}
        self.fitted = False

    def drop_unwanted_columns(self, df):
        cols = [c for c in self.drop_columns if c in df.columns]
        if cols:
            df = df.drop(columns=cols)
            logger.info(f"Dropped columns: {cols}")
        return df

    def coerce_numeric(self, df):
        for col in df.columns:
            if df[col].dtype == "object" and col != self.target_column:
                converted = pd.to_numeric(df[col], errors="coerce")
                if converted.notna().mean() > 0.9:
                    df[col] = converted
                    logger.info(f"Coerced column '{col}' to numeric")
        return df

    def encode_target(self, df, fitting):
        if not self.encode_target_flag or self.target_column not in df.columns:
            return df
        if df[self.target_column].dtype == "object":
            if fitting:
                self.target_encoder = LabelEncoder()
                df[self.target_column] = self.target_encoder.fit_transform(df[self.target_column])
                logger.info(f"Encoded target '{self.target_column}', classes: {list(self.target_encoder.classes_)}")
            else:
                df[self.target_column] = self.target_encoder.transform(df[self.target_column])
        return df

    def handle_missing_values(self, df, fitting):
        strategy = self.missing_strategy
        if strategy == "drop":
            before = len(df)
            df = df.dropna()
            logger.info(f"Dropped {before - len(df)} rows with missing values")
            return df

        if fitting:
            self.impute_values = {}
            for col in df.select_dtypes(include="number").columns:
                self.impute_values[col] = (
                    df[col].median() if strategy == "median" else df[col].mean()
                )
            for col in df.select_dtypes(include="object").columns:
                if col != self.target_column:
                    self.impute_values[col] = df[col].mode().iloc[0]

        for col, value in self.impute_values.items():
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(value)
                logger.info(f"Filled missing in '{col}' with {value}")
        return df

    def apply_log_transform(self, df, fitting):
        mode = self.log_config.get("mode", "off")
        if mode == "off":
            return df

        if fitting:
            if mode == "manual":
                self.log_columns = self.log_config.get("columns", [])
            else:  # auto
                threshold = self.log_config.get("skew_threshold", 0.75)
                self.log_columns = []
                numeric = df.select_dtypes(include="number").drop(
                    columns=[self.target_column], errors="ignore")
                for col in numeric.columns:
                    if (df[col] > 0).all() and abs(df[col].skew()) > threshold:
                        self.log_columns.append(col)
                logger.info(f"Auto-detected log-transform columns: {self.log_columns}")

        for col in self.log_columns:
            if col in df.columns:
                df[col] = np.log1p(df[col])
                logger.info(f"Applied log transform to '{col}'")
        return df

    def handle_outliers(self, df, fitting):
        # Outlier removal only runs while fitting (training data). Rows are
        # never dropped from data we are scoring.
        if not self.outlier_config.get("handle", False) or not fitting:
            return df

        multiplier = self.outlier_config.get("multiplier", 1.5)
        min_unique = self.outlier_config.get("min_unique", 10)
        cols = self.outlier_config.get("columns", [])
        if not cols:
            numeric = df.select_dtypes(include="number").drop(
                columns=[self.target_column], errors="ignore").columns.tolist()
            cols = [c for c in numeric if df[c].nunique() >= min_unique]
            skipped = [c for c in numeric if c not in cols]
            if skipped:
                logger.info(f"Skipped low-cardinality columns for outliers: {skipped}")
            logger.info(f"No outlier columns given; using continuous numeric: {cols}")
        else:
            cols = [c for c in cols if c in df.columns
                    and pd.api.types.is_numeric_dtype(df[c])]
            logger.info(f"Applying outlier removal to columns: {cols}")

        before = len(df)
        for col in cols:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            low, high = q1 - multiplier * iqr, q3 + multiplier * iqr
            df = df[(df[col] >= low) & (df[col] <= high)]
        logger.info(f"Outlier removal dropped {before - len(df)} rows")
        return df

    def encode_features(self, df):
        cat_cols = [c for c in df.select_dtypes(include="object").columns
                    if c != self.target_column]
        if not cat_cols:
            return df

        if self.encoding == "label":
            for col in cat_cols:
                df[col] = LabelEncoder().fit_transform(df[col])
            logger.info(f"Label encoded columns: {cat_cols}")
        else:
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
            logger.info(f"One-hot encoded columns: {cat_cols}")
        return df

    def align_columns(self, df, fitting):
        # After one-hot encoding, train and test can end up with different
        # columns. Lock the schema on fit, then align test data to it.
        if fitting:
            self.feature_columns = [c for c in df.columns if c != self.target_column]
            return df
        target = df[self.target_column] if self.target_column in df.columns else None
        features = df.reindex(columns=self.feature_columns, fill_value=0)
        if target is not None:
            features[self.target_column] = target.values
        logger.info("Aligned test columns to training schema")
        return features

    def scale_features(self, df, fitting):
        if self.scaling == "none":
            return df

        scalers = {"standard": StandardScaler, "minmax": MinMaxScaler, "normalize": Normalizer}
        scaler_cls = scalers.get(self.scaling)
        if scaler_cls is None:
            logger.warning(f"Unknown scaling '{self.scaling}', skipping")
            return df

        if fitting:
            self.scaled_columns = df.select_dtypes(include="number").drop(
                columns=[self.target_column], errors="ignore").columns.tolist()
            self.scaler = scaler_cls()
            df[self.scaled_columns] = self.scaler.fit_transform(df[self.scaled_columns])
            logger.info(f"Fitted {self.scaling} scaling on {len(self.scaled_columns)} columns")
        else:
            df[self.scaled_columns] = self.scaler.transform(df[self.scaled_columns])
            logger.info(f"Applied {self.scaling} scaling to test data")
        return df

    def fit_transform(self, df):
        df = df.copy()
        df = self.drop_unwanted_columns(df)
        df = self.coerce_numeric(df)
        df = self.encode_target(df, fitting=True)
        df = self.handle_missing_values(df, fitting=True)
        df = self.apply_log_transform(df, fitting=True)
        df = self.handle_outliers(df, fitting=True)
        df = self.encode_features(df)
        df = self.align_columns(df, fitting=True)
        df = self.scale_features(df, fitting=True)
        self.fitted = True
        logger.info(f"fit_transform complete: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    def transform(self, df):
        if not self.fitted:
            raise RuntimeError("Call fit_transform on training data before transform.")
        df = df.copy()
        df = self.drop_unwanted_columns(df)
        df = self.coerce_numeric(df)
        df = self.encode_target(df, fitting=False)
        df = self.handle_missing_values(df, fitting=False)
        df = self.apply_log_transform(df, fitting=False)
        df = self.handle_outliers(df, fitting=False)
        df = self.encode_features(df)
        df = self.align_columns(df, fitting=False)
        df = self.scale_features(df, fitting=False)
        logger.info(f"transform complete: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    def show_overview(self, df):
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Missing values total: {df.isnull().sum().sum()}")
        logger.info(f"Dtypes: {df.dtypes.value_counts().to_dict()}")