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
    PREPROCESS
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

        # Separate features and target
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        if PREPROCESS:
            # Standardize the features
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_seed)
        
        return X_train, X_test, y_train, y_test