"""
Utility / exploratory helpers.

These are exploration and inspection helpers - useful when investigating a
dataset, but deliberately kept SEPARATE from the preprocessing pipeline.
Preprocessing transforms data; these just display it. Keeping them apart
keeps the Preprocessor focused on a single responsibility.
"""

import logging
import yaml


logger = logging.getLogger(__name__)


def show_summary(df):
    """Print structure and dtypes of the DataFrame."""
    logger.info("Showing DataFrame summary")
    print("DataFrame summary:")
    df.info()


def show_head(df, n=5):
    """Print the first n rows."""
    logger.info(f"Showing first {n} rows")
    print(df.head(n))


def show_missing_values(df):
    """Print columns that contain missing values."""
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    logger.info("Showing missing values")
    if missing.empty:
        print("No missing values.")
    else:
        print("Missing values per column:")
        print(missing)


def show_value_counts(df, column):
    """Print the value distribution of a single column."""
    if column not in df.columns:
        logger.warning(f"Column '{column}' not found")
        return
    logger.info(f"Showing value counts for '{column}'")
    print(f"Value counts for '{column}':")
    print(df[column].value_counts())


def show_descriptive_statistics(df):
    """Print descriptive statistics for numeric columns."""
    logger.info("Showing descriptive statistics")
    print("Descriptive statistics:")
    print(df.describe())


def show_correlation_matrix(df):
    """Print the correlation matrix for numeric columns only."""
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        logger.warning("No numeric columns for correlation")
        return
    logger.info("Showing correlation matrix")
    print("Correlation matrix:")
    print(numeric_df.corr())


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)