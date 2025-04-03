import os
import pandas as pd
import numpy as np
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score,
                             roc_curve, auc)
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker # Import ticker

# Assuming logger and config imports are correct and OUTPUT_PATH is defined in config
from Utils.logger import get_logger
from Utils.config import (
    RANDOM_SEED,
    JOBS,
    OUTPUT_PATH
 )

logger = get_logger()

class Analysis:
    def __init__(self, model, X_test, y_test, y_pred):
        self.X_test = X_test
        self.model = model
        self.y_test = y_test
        self.y_pred = y_pred # Predictions at default 0.5 threshold

    def metrics(self):
        # Metrics at default threshold
        try:
            accuracy = accuracy_score(self.y_test, self.y_pred)
            precision = precision_score(self.y_test, self.y_pred, zero_division=0) # type: ignore[arg-type]
            recall = recall_score(self.y_test, self.y_pred, zero_division=0)    # type: ignore[arg-type]
            f1 = f1_score(self.y_test, self.y_pred, zero_division=0)       # type: ignore[arg-type]
            logger.info(f"Metrics (Default Threshold ~0.5): Acc={accuracy:.4f}, Prc={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}")
        except Exception as e:
            logger.error(f"Error calculating classification metrics: {e}")

    def confusion_matrix(self):
        # Confusion Matrix at default threshold
        try:
            cm = confusion_matrix(self.y_test, self.y_pred)
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0 # TNR for class 0
            recall_tpr = recall_score(self.y_test, self.y_pred, zero_division=0) # TPR for class 1
            logger.info(f"Confusion Matrix (Default Threshold ~0.5): [[{tn}, {fp}], [{fn}, {tp}]] | TNR(0)={specificity:.4f}, TPR(1)={recall_tpr:.4f}")
            return cm
        except Exception as e:
            logger.error(f"Error calculating confusion matrix: {e}")
            return None

    def feature_importance(self):
        # Feature importance calculation (no changes needed here)
        if not isinstance(self.X_test, pd.DataFrame) or not hasattr(self.X_test, 'columns'):
             logger.error("Feature importance skipped: X_test missing columns.")
             return
        try:
            # ... (rest of feature importance code remains the same) ...
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
            if OUTPUT_PATH:
                 # ... (saving code remains the same) ...
                 save_path = os.path.join(OUTPUT_PATH, "feature_importance.csv")
                 perm_importance_df.to_csv(save_path, index=False)
                 logger.info(f"Feature importance saved: {save_path}")
            else:
                 logger.warning("Feature importance not saved (OUTPUT_PATH undefined/inaccessible).")

        except Exception as e:
             logger.error(f"Error calculating feature importance: {e}")


    def roc_curve_analysis(self):
        """
        Calculates ROC curve, AUC, finds specific operating points based on project goals,
        and plots the results.
        """
        if not hasattr(self.model, "predict_proba"):
            logger.error("ROC analysis skipped: Model lacks predict_proba.")
            return

        try:
            # Assuming positive class (good outcome) is 1
            y_scores = self.model.predict_proba(self.X_test)[:, 1]
        except Exception as e:
            logger.error(f"Error getting model probabilities: {e}")
            return

        try:
            # Calculate base ROC curve elements
            # Note: fpr = False Positive Rate for class 1 (good outcome)
            #       tpr = True Positive Rate for class 1 (good outcome)
            fpr, tpr, thresholds = roc_curve(self.y_test, y_scores)
            roc_auc = auc(fpr, tpr)
            logger.info(f"Overall ROC AUC: {roc_auc:.4f}") # Log overall performance

            # --- Find Operating Point 1: Specificity(poor) = 100% (TPR(good) = 1.0) ---
            logger.info("-" * 20)
            logger.info("Finding Operating Point for Specificity(poor) = 1.0 (i.e., TPR(good) = 1.0):")
            indices_tpr_one = np.where(np.isclose(tpr, 1.0))[0] # Use isclose for float comparison
            threshold_for_tpr_one = None
            fpr_at_tpr_one = 1.0 # Default to worst case FPR

            if len(indices_tpr_one) > 0:
                # Find the index within this group that has the LOWEST FPR
                best_idx_tpr_one = indices_tpr_one[np.argmin(fpr[indices_tpr_one])]
                fpr_at_tpr_one = fpr[best_idx_tpr_one]

                # Find corresponding threshold (thresholds[i-1] corresponds to fpr[i], tpr[i])
                if best_idx_tpr_one > 0 and best_idx_tpr_one <= len(thresholds):
                    threshold_for_tpr_one = thresholds[best_idx_tpr_one - 1]
                elif best_idx_tpr_one == 0 and len(thresholds) > 0: # Edge case if tpr[0] is 1.0
                     threshold_for_tpr_one = thresholds[0] # Might need adjustment based on roc_curve behavior
                elif len(thresholds)==0: # No thresholds returned
                     threshold_for_tpr_one = None
                else: # best_idx_tpr_one > len(thresholds) - handle potential off-by-one from roc_curve
                     threshold_for_tpr_one = thresholds[-1]


                logger.info(f"  >> Achieved TPR(good) = 1.0")
                logger.info(f"  >> Lowest corresponding FPR(good) = {fpr_at_tpr_one:.4f} (Specificity(good) = {(1-fpr_at_tpr_one):.4f})")
                if threshold_for_tpr_one is not None:
                    logger.info(f"  >> Corresponding Threshold ~= {threshold_for_tpr_one:.4f}")
                else:
                    logger.warning("  >> Could not determine specific threshold reliably (likely requires very low score).")
            else:
                logger.warning("  >> Could not find operating point where TPR(good) strictly equals 1.0.")


            # --- Find Operating Point 2: Specificity(good) >= 95% (FPR(good) <= 0.05) ---
            logger.info("-" * 20)
            logger.info("Finding Operating Point for Specificity(good) >= 0.95 (i.e., FPR(good) <= 0.05):")
            target_fpr = 0.05
            indices_low_fpr = np.where(fpr <= target_fpr)[0]
            threshold_for_low_fpr = None
            tpr_at_low_fpr = 0.0
            actual_fpr_at_low_fpr = target_fpr # Start assuming limit is hit

            if len(indices_low_fpr) > 0:
                # Find the index within this low_fpr group that gives the highest TPR
                best_idx_low_fpr = indices_low_fpr[np.argmax(tpr[indices_low_fpr])]
                tpr_at_low_fpr = tpr[best_idx_low_fpr]
                actual_fpr_at_low_fpr = fpr[best_idx_low_fpr] # The actual FPR achieved

                # Find corresponding threshold
                if best_idx_low_fpr > 0 and best_idx_low_fpr <= len(thresholds):
                    threshold_for_low_fpr = thresholds[best_idx_low_fpr - 1]
                elif best_idx_low_fpr == 0 and len(thresholds) > 0: # If (0,0) point meets criteria
                     threshold_for_low_fpr = thresholds[0]
                elif len(thresholds)==0:
                     threshold_for_low_fpr = None
                else: # best_idx_low_fpr > len(thresholds)
                     threshold_for_low_fpr = thresholds[-1]


                logger.info(f"  >> Target FPR(good) <= {target_fpr:.3f} achieved.")
                logger.info(f"  >> Highest corresponding TPR(good) = {tpr_at_low_fpr:.4f}")
                logger.info(f"  >> Actual FPR(good) at this point = {actual_fpr_at_low_fpr:.4f} (Specificity(good) = {(1-actual_fpr_at_low_fpr):.4f})")
                if threshold_for_low_fpr is not None:
                    logger.info(f"  >> Corresponding Threshold ~= {threshold_for_low_fpr:.4f}")
                else:
                    logger.warning("  >> Could not determine specific threshold reliably.")

            else:
                logger.warning(f"  >> Could NOT find any operating point with FPR(good) <= {target_fpr:.3f}.")
            logger.info("-" * 20)


            # --- Plotting ---
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

            # Highlight Operating Point 1 (TPR=1) if found reliably
            if len(indices_tpr_one) > 0 and threshold_for_tpr_one is not None:
                 ax.scatter(fpr_at_tpr_one, 1.0, s=100, facecolors='none', edgecolors='green', zorder=5, linewidth=1.5,
                           label=f'Spec(poor)=1 (Th~{threshold_for_tpr_one:.2f})')

            # Highlight Operating Point 2 (FPR<=0.05) if found reliably
            if len(indices_low_fpr) > 0 and threshold_for_low_fpr is not None:
                 ax.scatter(actual_fpr_at_low_fpr, tpr_at_low_fpr, s=100, facecolors='none', edgecolors='blue', zorder=5, linewidth=1.5,
                           label=f'Spec(good)>=0.95 (Th~{threshold_for_low_fpr:.2f})')

            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ticks = np.arange(0.0, 1.1, 0.1)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            # ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%0.1f'))
            # ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%0.1f'))
            ax.set_xlabel('False Positive Rate (1 - Specificity for Good Outcome)') # Clarified label
            ax.set_ylabel('True Positive Rate (Sensitivity for Good Outcome)') # Clarified label
            ax.set_title('ROC Curve Analysis')
            ax.legend(loc="lower right")
            ax.grid(True)

            # --- Save or Show Plot ---
            if OUTPUT_PATH:
                # ... (Saving code remains the same) ...
                roc_save_path = os.path.join(OUTPUT_PATH, "roc_curve.png")
                fig.savefig(roc_save_path, dpi=150, bbox_inches='tight')
                logger.info(f"ROC curve plot saved: {roc_save_path}")
            else:
                 plt.show()

            plt.close(fig)

        except Exception as e:
            logger.error(f"Error during ROC curve analysis: {e}")
