import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random

from Utils.logger import get_logger

# Assuming these constants are defined in your Utils.config file
from Utils.config import (
    DATA_PATH,
    FILES_USED,
    TARGET_COLUMN,
    ALL_FEATURES,
    FEATURES_TO_KEEP,
    SEED
)

logger = get_logger()

class DataLoader:
    def __init__(self):
        self.data_folder = DATA_PATH
        self.random_seed = SEED
        self.files_used = FILES_USED
        self.dataframes = {}
        self.target_column = TARGET_COLUMN
        self.features_to_keep = FEATURES_TO_KEEP # Store the list

    def load_data(self):
        # --- Load data (ensure this part works correctly) ---
        loaded_files_count = 0
        if not os.path.isdir(self.data_folder):
            print(f"Error: Data path not found or not a directory: '{self.data_folder}'")
            return False # Indicate failure

        for file in os.listdir(self.data_folder):
            if file.lower().endswith(('.xlsx', '.xls')):
                file_path = os.path.join(self.data_folder, file)
                try:
                    df_temp = pd.read_excel(file_path)
                    if not df_temp.empty:
                        self.dataframes[file] = df_temp
                        loaded_files_count += 1
                    # else: # Optional: log skipping empty file
                    #     print(f"Warning: Skipping empty file: {file}")
                except Exception as e:
                    print(f"Error occurred while loading {file}: {e}")

        if loaded_files_count == 0:
             print(f"Warning: No valid Excel files were loaded from '{self.data_folder}'.")
             return False
        return True


    def preprocess_data(self):
        df = None
        available_files = list(self.dataframes.keys())
        num_files = len(available_files)

        if not self.dataframes: # Check if load_data actually loaded anything
             raise ValueError("Cannot preprocess data: No dataframes were loaded. Run load_data() first.")

        # --- Select DataFrame(s) based on FILES_USED ---
        if self.files_used == '1':
            if num_files >= 1:
                file_key = available_files[0]
                df = self.dataframes[file_key].copy() # Use copy
            else:
                raise ValueError("Config error: FILES_USED='1', but no files loaded.")
        elif self.files_used == '2':
            if num_files >= 2:
                file_key = available_files[1]
                df = self.dataframes[file_key].copy() # Use copy
            else:
                raise ValueError(f"Config error: FILES_USED='2', but only {num_files} file(s) available.")
        else:
             raise ValueError(f"Invalid value for FILES_USED: '{self.files_used}'.")

        if df is None:
             raise RuntimeError("Internal error: DataFrame 'df' not assigned.")
        if df.empty:
            raise ValueError("Preprocessing error: Selected DataFrame is empty.")

        # Separate target first
        y = df[self.target_column]
        # Get all potential feature columns (excluding target)
        x_features = df.drop(columns=[self.target_column]).columns.tolist()

        # Feature selection step
        if not ALL_FEATURES:
            X = df[self.features_to_keep]
            print(f"Applied feature selection. Kept {len(self.features_to_keep)} features.")
        else:
            X = df[x_features]
            print(f"All features selected. Total features: {len(x_features)}")

        """
        # --- Preprocessing (Scaling) ---
        X_processed = X.copy() # Start with selected features, use copy
        if PREPROCESS:
            numeric_cols = X_processed.select_dtypes(include='number').columns
            if not numeric_cols.empty:
                scaler = StandardScaler()
                X_processed[numeric_cols] = scaler.fit_transform(X_processed[numeric_cols])
                print("Applied StandardScaler to selected numeric features.")
            else:
                 print("Warning: No numeric columns found to scale in the selected features.")
        else:
            print("Preprocessing (StandardScaler) skipped as PREPROCESS is False.")
        # -----------------------------
        """


        # --- Split the data ---
        # Use stratify for potentially imbalanced classification data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_seed)
            print(f"Data split complete. Shapes: X_train={X_train.shape}, X_test={X_test.shape}")
            logger.info(f"Random seed used for splitting: {self.random_seed}")
            return X_train, X_test, y_train, y_test
        except Exception as e:
             raise RuntimeError(f"Error during train_test_split: {e}")
