import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Utils.config import (
    DATA_PATH,
    RANDOM_SEED,
    FILES_USED,
    TARGET_COLUMN,
)


class DataLoader:
    def __init__(self):
        self.data_folder = DATA_PATH
        self.random_seed = RANDOM_SEED
        self.files_used = FILES_USED
        self.dataframes = {}
        self.target_column = TARGET_COLUMN

    def load_data(self):
        for file in os.listdir(self.data_folder):
            if file.endswith('.xlsx') or file.endswith('.xls'):
                file_path = os.path.join(self.data_folder, file)
                df = pd.read_excel(file_path)
                self.dataframes[file] = df

    def preprocess_data(self):
        if self.files_used == '1':
            # Use only the first file
            file = list(self.dataframes.keys())[0]
            df = self.dataframes[file]
        elif self.files_used == '2':
            # Use only the second file
            file = list(self.dataframes.keys())[1]
            df = self.dataframes[file]
        elif self.files_used == '3':
            # Use both files
            if len(self.dataframes) > 1:
                # Concatenate the dataframes
                df = pd.concat(self.dataframes.values(), ignore_index=True)

        # Transform the target variable to binary
        df[self.target_column] = df[self.target_column].apply(
        lambda x: 1 if x in [1, 2] else 0)

        # Separate features and target
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]





        # Start with the full preprocessing pipeline
        return 0