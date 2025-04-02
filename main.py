import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
import os 
import pandas as pd
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

from Utils import (
    logger,
    dataloader
)

from Utils.config import (
    DATA_PATH,
    RANDOM_SEED,
    FILES_USED,
    TARGET_COLUMN,
    DEVICE,
)


if __name__ == "__main__":

    # Initialize logger
    logger.initialize_logger()
    logger = logger.get_logger()

    logger.info("-" * 50)
    logger.info("Executing main")
    logger.info("-" * 50)

    # Load data
    data_loader = dataloader.DataLoader()
    data_loader.load_data()
    X_train, X_test, y_train, y_test = data_loader.preprocess_data()
    logger.info("Data loaded and preprocessed")
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}")
    logger.info(f"y_train shape: {y_train.shape}")
    logger.info(f"y_test shape: {y_test.shape}")
    logger.info("-" * 50)

    # Defining the model randomforest classifier
    model = RandomForestClassifier(random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    logger.info("Model trained")
    logger.info("-" * 50)
    
    # Predicting the test set
    y_pred = model.predict(X_test)
    logger.info("Model predicted")
    logger.info("-" * 50)

