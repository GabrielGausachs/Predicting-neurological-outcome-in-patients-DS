from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from Utils.logger import initialize_logger, get_logger

from Utils.config import (
    DATA_PATH,
    RANDOM_SEED,
    FILES_USED,
    TARGET_COLUMN,
    DEVICE,
    JOBS,
    OUTPUT_PATH,
    N_ESTIMATORS,
    MAX_FEATURES,
    MAX_DEPTH,
 )

logger = get_logger()

def train(X_train, y_train, X_test, y_test):
    rf_model = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=JOBS, oob_score=True)
    param_grid = {
        'n_estimators': N_ESTIMATORS,  # Number of trees in the forest
        'max_features': MAX_FEATURES,    # Number of features to consider when looking for the best split
        'max_depth': MAX_DEPTH,          # Maximum depth of the tree                                                       
    }

    grid_search = GridSearchCV(
        estimator=rf_model,        # The model to tune
        param_grid=param_grid,     # Dictionary of parameters to try
        cv=5,                      # Number of cross-validation folds
        scoring='accuracy',        # Metric to optimize
        n_jobs=JOBS,                 # Use all available CPU cores
        verbose=2                  # How much information to display (higher is more)
    )

    logger.info("Starting Grid Search for Hyperparameter Tuning...")
    grid_search.fit(X_train, y_train)
    logger.info("Grid Search Complete.")

    logger.info("-" * 50)
    logger.info("Best Parameters found by Grid Search:")
    logger.info(grid_search.best_params_)
    logger.info(f"Best cross-validation accuracy score: {grid_search.best_score_:.4f}")

    best_rf_model = grid_search.best_estimator_

    return best_rf_model, grid_search.best_params_

