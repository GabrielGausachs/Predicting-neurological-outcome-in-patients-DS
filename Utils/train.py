from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from Utils.logger import initialize_logger, get_logger
from sklearn.metrics import (accuracy_score, make_scorer, precision_score, 
                            recall_score, f1_score)

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

def train_rf(X_train, y_train):
    rf_model = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=JOBS,bootstrap=False)
    param_grid = {
        'n_estimators': N_ESTIMATORS,  # Number of trees in the forest
        'max_features': MAX_FEATURES,    # Number of features to consider when looking for the best split
        'max_depth': MAX_DEPTH,          # Maximum depth of the tree                                                       
    }

    grid_search = GridSearchCV(
        estimator=rf_model,      
        param_grid=param_grid,
        scoring='roc_auc', 
        cv=5,       
        n_jobs=-1,
        verbose=2,
        return_train_score=True, # Return training scores
    )

    logger.info(f"Starting Grid Search for Hyperparameter Tuning...")
    grid_search.fit(X_train, y_train)
    logger.info("Grid Search Complete.")

    logger.info("-" * 50)
    logger.info("Best Parameters found by Grid Search:")
    logger.info(grid_search.best_params_)
    logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    best_rf_model = grid_search.best_estimator_

    return best_rf_model, grid_search.best_params_

