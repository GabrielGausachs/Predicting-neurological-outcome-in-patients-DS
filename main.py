import numpy as np
import os
import pickle

from Utils import (
    logger,
    dataloader,
    analysis,
    train
)

from Utils.config import (
    MODELS_PATH,
    MODEL_NAME,
    TRAINING
)

if __name__ == "__main__":

    # Initialize logger
    logger.initialize_logger()
    logger = logger.get_logger()

    logger.info("-" * 50)
    logger.info("Executing main")
    logger.info("-" * 50)
    logger.info(f"Executing for {MODEL_NAME} case")

    # Load data
    data_loader = dataloader.DataLoader()
    data_loader.load_data()
    # Keep data types consistent where needed
    X_train_df, X_test_df, y_train_arr, y_test_arr = data_loader.preprocess_data()

    # Convert y arrays if they aren't already (usually returned as Series/arrays)
    y_train_arr = np.array(y_train_arr)
    y_test_arr = np.array(y_test_arr)

    logger.info("Data loaded and preprocessed")
    logger.info(f"X_train shape: {X_train_df.shape}") # Use DataFrame shape
    logger.info(f"X_test shape: {X_test_df.shape}")
    logger.info(f"y_train shape: {y_train_arr.shape}")
    logger.info(f"y_test shape: {y_test_arr.shape}")
    logger.info("-" * 50)

    if TRAINING:
        # Load and train the model using the DataFrame for X_train
        logger.info("Training the model")
        best_rf_model, best_params = train.train_rf(X_train_df, y_train_arr)
        # ------------------------------------------------
        logger.info("Model trained")
        logger.info("-" * 50)

        # Save the model
        logger.info("Saving the model")
        model_path = os.path.join(MODELS_PATH, f"{MODEL_NAME}.pkl")
        with open(model_path, 'wb') as model_file:
            pickle.dump(best_rf_model, model_file)
        logger.info(f"Model saved at: {model_path}")

    else:
        # Load the model
        logger.info("Loading the model")
        model_path = os.path.join(MODELS_PATH, f"{MODEL_NAME}.pkl")
        with open(model_path, 'rb') as model_file:
            best_rf_model = pickle.load(model_file)
        logger.info(f"Model loaded from: {model_path}")

    # Predict the test set (predict usually works fine with DataFrame input)
    y_pred_arr = best_rf_model.predict(X_test_df)
    logger.info("Model predicted")
    logger.info("-" * 50)

    # Analyze the model - passing DataFrame X_test_df is correct here
    logger.info("Analyzing the model")
    analysis_model = analysis.Analysis(best_rf_model, X_test_df, y_test_arr, y_pred_arr)
    analysis_model.metrics()
    analysis_model.confusion_matrix()

    # Feature importance analysis
    #analysis_model.feature_importance()

    # ROC curve analysis
    thres_poor,thres_good = analysis_model.roc_curve_analysis()

    # Calculate the accuracy for both thresholds
    acc = analysis_model.calculate_accuracy(thres_good)
    logger.info(f"Accruacy for threshold {thres_good}: {acc:.4f}")
    acc = analysis_model.calculate_accuracy(thres_poor)
    logger.info(f"Accruacy for threshold {thres_poor}: {acc:.4f}")



