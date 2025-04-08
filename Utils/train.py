from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from Utils.logger import get_logger

from Utils.config import (
    RANDOM_SEED,
    JOBS,
    N_ESTIMATORS,
    MAX_FEATURES,
    MAX_DEPTH,
 )

logger = get_logger()

def train_rf(X_train, y_train):

    # Create a RandomForestClassifier model
    rf_model = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=JOBS, bootstrap=True)
    param_grid = {
        'n_estimators': N_ESTIMATORS,  # Number of trees in the forest
        'max_features': MAX_FEATURES,    # Number of features to consider when looking for the best split
        'max_depth': MAX_DEPTH}          # Maximum depth of the tree

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=7,
        n_jobs=JOBS,
        verbose=2,
        return_train_score=True)

    # Fit the model
    logger.info("Starting Grid Search for Hyperparameter Tuning...")
    grid_search.fit(X_train, y_train)
    logger.info("Grid Search Complete.")

    # Get the best parameters and the best model
    logger.info("-" * 50)
    logger.info("Best Parameters found by Grid Search:")
    logger.info(grid_search.best_params_)
    logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    best_rf_model = grid_search.best_estimator_

    return best_rf_model, grid_search.best_params_
