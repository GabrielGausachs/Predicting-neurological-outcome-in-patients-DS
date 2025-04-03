import os
import pandas as pd
import numpy as np
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score,
                             roc_curve, auc)
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# Assuming logger and config imports are correct and OUTPUT_PATH is defined in config
from Utils.logger import get_logger
from Utils.config import (
    RANDOM_SEED,
    JOBS,
    OUTPUT_PATH # Pyright needs to find this defined in the actual config file
 )

logger = get_logger()

class Analysis:
    def __init__(self, model, X_test, y_test, y_pred):
        self.X_test = X_test
        self.model = model
        self.y_test = y_test
        self.y_pred = y_pred

    def metrics(self):
        try:
            accuracy = accuracy_score(self.y_test, self.y_pred)
            # Add type ignore comments for Pyright
            precision = precision_score(self.y_test, self.y_pred, zero_division=0) # type: ignore[arg-type]
            recall = recall_score(self.y_test, self.y_pred, zero_division=0)    # type: ignore[arg-type]
            f1 = f1_score(self.y_test, self.y_pred, zero_division=0)       # type: ignore[arg-type]
            logger.info(f"Metrics: Acc={accuracy:.4f}, Prc={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}")
        except Exception as e:
            logger.error(f"Error calculating classification metrics: {e}")

    def confusion_matrix(self):
        try:
            cm = confusion_matrix(self.y_test, self.y_pred)
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            logger.info(f"Confusion Matrix: [[{tn}, {fp}], [{fn}, {tp}]] | TNR={specificity:.4f}")
            return cm
        except Exception as e:
            logger.error(f"Error calculating confusion matrix: {e}")
            return None

    def feature_importance(self):
        # Check X_test type without logging for brevity, error out if needed
        if not isinstance(self.X_test, pd.DataFrame) or not hasattr(self.X_test, 'columns'):
             logger.error("Feature importance skipped: X_test missing columns.")
             return

        try:
            result = permutation_importance(self.model, self.X_test, self.y_test, n_repeats=10, random_state=RANDOM_SEED, n_jobs=JOBS)
            importances_perm_mean = result["importances_mean"]
            importances_perm_std = result["importances_std"]
            feature_names = self.X_test.columns.tolist()

            perm_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance_Perm_Mean': importances_perm_mean,
                'Importance_Perm_Std': importances_perm_std
            })
            perm_importance_df = perm_importance_df.sort_values(by='Importance_Perm_Mean', ascending=False)

            # Use OUTPUT_PATH - this must be defined in Utils.config.py
            if OUTPUT_PATH:
                try:
                    if not os.path.exists(OUTPUT_PATH):
                        os.makedirs(OUTPUT_PATH) # Create dir if needed
                    save_path = os.path.join(OUTPUT_PATH, "feature_importance.csv")
                    perm_importance_df.to_csv(save_path, index=False)
                    logger.info(f"Feature importance saved: {save_path}")
                except OSError as e:
                     logger.error(f"Could not create/access output directory {OUTPUT_PATH}: {e}")
                except Exception as e: # Catch other potential save errors
                     logger.error(f"Error saving feature importance to {OUTPUT_PATH}: {e}")
            else:
                 logger.warning("Feature importance not saved (OUTPUT_PATH undefined/inaccessible).")

        except Exception as e:
             logger.error(f"Error calculating feature importance: {e}")


    def roc_curve_analysis(self): # Ensure main.py calls this name
        if not hasattr(self.model, "predict_proba"):
            logger.error("ROC analysis skipped: Model lacks predict_proba.")
            return

        try:
            y_scores = self.model.predict_proba(self.X_test)[:, 1]
        except Exception as e:
            logger.error(f"Error getting model probabilities: {e}")
            return

        try:
            fpr, tpr, thresholds = roc_curve(self.y_test, y_scores)
            roc_auc = auc(fpr, tpr)

            # --- Find Threshold ---
            indices_fpr_zero = np.where(fpr == 0.0)[0]
            candidate_thresholds = []
            if len(indices_fpr_zero) > 0:
                for idx in indices_fpr_zero:
                    if tpr[idx] >= 0.95:
                         # Standard threshold alignment: thresholds[i] relates to fpr[i+1], tpr[i+1]
                         # So index 'idx' in fpr/tpr relates to thresholds[idx-1]
                         if idx > 0 and idx <= len(thresholds):
                              candidate_thresholds.append(thresholds[idx-1])

            candidate_thresholds = sorted(list(np.unique([th for th in candidate_thresholds if np.isfinite(th)])))
            best_candidate_threshold = None
            if candidate_thresholds:
                best_candidate_threshold = max(candidate_thresholds)
                logger.info(f"Highest Threshold (FPR=0, TPR>=0.95): {best_candidate_threshold:.4f}")
            # --- End Find Threshold ---

            # --- Plotting ---
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

            if best_candidate_threshold is not None:
                 best_idx = -1
                 for i in range(1, len(fpr)):
                     if i <= len(thresholds) and np.isclose(thresholds[i-1], best_candidate_threshold):
                          best_idx = i
                          break
                 if best_idx != -1 and best_idx < len(fpr):
                      plt.scatter(fpr[best_idx], tpr[best_idx], s=80,
                                  facecolors='none', edgecolors='red', zorder=5)

            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc="lower right")
            plt.grid(True)

            # Use OUTPUT_PATH - this must be defined in Utils.config.py
            if OUTPUT_PATH:
                try:
                    if not os.path.exists(OUTPUT_PATH):
                         os.makedirs(OUTPUT_PATH) # Create dir if needed
                    roc_save_path = os.path.join(OUTPUT_PATH, "roc_curve.png")
                    plt.savefig(roc_save_path, dpi=150, bbox_inches='tight')
                    logger.info(f"ROC curve plot saved: {roc_save_path}")
                except OSError as e:
                     logger.error(f"Could not create/access output directory {OUTPUT_PATH}: {e}")
                except Exception as e:
                    logger.error(f"Error saving ROC curve plot: {e}")

            else:
                 plt.show()

            plt.close()
            logger.info(f"ROC AUC: {roc_auc:.4f}")

        except Exception as e:
            logger.error(f"Error during ROC curve analysis: {e}")
