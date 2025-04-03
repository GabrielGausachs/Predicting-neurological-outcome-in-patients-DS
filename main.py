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
    dataloader,
    analysis,
    train
)

from Utils.config import (
    RANDOM_SEED,
    JOBS,
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
    
    # Load and train the RandomForestClassifier model
    logger.info("Training the model")
    best_rf_model,best_params = train.train_rf(X_train, y_train)
    logger.info("Model trained")
    logger.info("-" * 50)
    
    # Predict the test set
    y_pred = best_rf_model.predict(X_test)
    logger.info("Model predicted")
    logger.info("-" * 50)

    # Analyze the model
    logger.info("Analyzing the model")
    analysis_model = analysis.Analysis(best_rf_model,X_test,y_test, y_pred)
    analysis_model.metrics()
    analysis_model.confusion_matrix()
    analysis_model.roc_curve()


