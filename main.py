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

    # Load the model
    model_rf = RandomForestClassifier(
        n_jobs=JOBS,
        oob_score=True, 
        bootstrap=True,
        n_estimators=100,
        max_features=15,
        random_state=RANDOM_SEED)
    
    # Train the model
    logger.info("Training the model")
    best_rf_model,best_params = train.train(X_train, y_train, X_test, y_test)
    logger.info("Model trained")
    logger.info("-" * 50)
    
    
    # Predict the test set
    y_pred = best_rf_model.predict(X_test)
    logger.info("Model predicted")
    logger.info("-" * 50)

    # Analyze the model
    logger.info("Analyzing the model")
    logger.info(f"Oob_Score: {model_rf.oob_score_}")
    analysis_model = analysis.Analysis(y_test, y_pred)
    analysis_model.classification_report()
    analysis_model.confusion_matrix()


