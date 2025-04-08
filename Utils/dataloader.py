import os
import pandas as pd
from sklearn.model_selection import train_test_split

from Utils.logger import get_logger

from Utils.config import (
    DATA_PATH,
    FILES_USED,
    TARGET_COLUMN,
    SEED
)

logger = get_logger()

class DataLoader:
    def __init__(self):
        self.data_folder = DATA_PATH
        self.seed = SEED
        self.files_used = FILES_USED
        self.dataframes = {}
        self.target_column = TARGET_COLUMN

    def load_data(self):

        # Load data
        loaded_files_count = 0
        if not os.path.isdir(self.data_folder):
            print(f"Error: Data path not found or not a directory: '{self.data_folder}'")
            return False

        for file in os.listdir(self.data_folder):
            if file.lower().endswith(('.xlsx', '.xls')):
                file_path = os.path.join(self.data_folder, file)
                try:
                    df_temp = pd.read_excel(file_path)
                    if not df_temp.empty:
                        self.dataframes[file] = df_temp
                        loaded_files_count += 1
                except Exception as e:
                    print(f"Error occurred while loading {file}: {e}")

        if loaded_files_count == 0:
             print(f"Warning: No valid Excel files were loaded from '{self.data_folder}'.")
             return False
        return True


    def preprocess_data(self):

        # Preprocess data
        df = None
        available_files = list(self.dataframes.keys())
        num_files = len(available_files)

        if not self.dataframes:
             raise ValueError("Cannot preprocess data: No dataframes were loaded. Run load_data() first.")

        # Select which file to use
        if self.files_used == '1':
            if num_files >= 1:
                file_key = available_files[0]
                df = self.dataframes[file_key].copy()
            else:
                raise ValueError("Config error: FILES_USED='1', but no files loaded.")
        elif self.files_used == '2':
            if num_files >= 2:
                file_key = available_files[1]
                df = self.dataframes[file_key].copy()
            else:
                raise ValueError(f"Config error: FILES_USED='2', but only {num_files} file(s) available.")
        else:
             raise ValueError(f"Invalid value for FILES_USED: '{self.files_used}'.")

        if df is None:
             raise RuntimeError("Internal error: DataFrame 'df' not assigned.")
        if df.empty:
            raise ValueError("Preprocessing error: Selected DataFrame is empty.")

        # Separate features and target variable
        y = df[self.target_column]
        x_features = df.drop(columns=[self.target_column]).columns.tolist()
        X = df[x_features]

        # Split the data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.seed)
            return X_train, X_test, y_train, y_test
        except Exception as e:
             raise RuntimeError(f"Error during train_test_split: {e}")
