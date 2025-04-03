import numpy as np

from Utils import (
    logger,
    dataloader,
    analysis,
    train
)

# Import OUTPUT_PATH to check if defined for feature importance saving
from Utils.config import OUTPUT_PATH

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

    # Load and train the model using the DataFrame for X_train
    logger.info("Training the model")
    # --- CHANGE HERE: Pass DataFrame to train_rf ---
    best_rf_model, best_params = train.train_rf(X_train_df, y_train_arr)
    # ------------------------------------------------
    logger.info("Model trained")
    logger.info("-" * 50)

    # Predict the test set (predict usually works fine with DataFrame input)
    y_pred_arr = best_rf_model.predict(X_test_df)
    logger.info("Model predicted")
    logger.info("-" * 50)

    # Analyze the model - passing DataFrame X_test_df is correct here
    logger.info("Analyzing the model")
    analysis_model = analysis.Analysis(best_rf_model, X_test_df, y_test_arr, y_pred_arr)
    analysis_model.metrics()
    analysis_model.confusion_matrix()

    # Optional feature importance call (needs X_test_df)
    if OUTPUT_PATH:
        analysis_model.feature_importance()
    else:
        logger.warning("Skipping feature importance analysis as OUTPUT_PATH is not defined in config.")

    # Call the correct analysis method
    analysis_model.roc_curve_analysis()
