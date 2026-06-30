import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_data(file_path, target_column):
    """
    Load data from a CSV file into a pandas DataFrame.

    Parameters:
    file_path (str): The path to the CSV file.
    target_column (str): The name of the target column.

    Returns:
    pd.DataFrame: A DataFrame containing the loaded data.
    """
    try:
        data = pd.read_csv(file_path,index_col=False)
    except FileNotFoundError:
        logger.error(f"The file {file_path} was not found.")
        raise
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        return None

    if data.empty:
         raise ValueError(f"The file {file_path} is empty.")
    if target_column not in data.columns:
        raise ValueError(f"The target column '{target_column}' is not present in the data.")
    if data[target_column].isnull().any():
        logger.warning(f"The target column '{target_column}' contains missing values.")
    
    logger.info(f'Loaded {len(data)} rows and {len(data.columns)} columns from {file_path}.')
    
    return data