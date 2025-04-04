import os
import pandas as pd
import numpy as np
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score,
                             roc_curve, auc)
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

from Utils.logger import get_logger
from Utils.config import (
    RANDOM_SEED,
    JOBS,
    OUTPUT_PATH,
    ALL_FEATURES,
    FILES_USED,
 )

logger = get_logger()

class Analysis:
    def __init__(self, model, X_test, y_test, y_pred):
        self.X_test = X_test
        self.model = model
        self.y_test = y_test
        self.y_pred = y_pred # Predictions at default 0.5 threshold

    def metrics(self):
        try:
            accuracy = accuracy_score(self.y_test, self.y_pred)
            precision_good = precision_score(self.y_test, self.y_pred, zero_division=0) # type: ignore[arg-type]
            recall_good = recall_score(self.y_test, self.y_pred, zero_division=0)    # type: ignore[arg-type] # = Spec(Poor)
            f1_good = f1_score(self.y_test, self.y_pred, zero_division=0)       # type: ignore[arg-type]
            # Calculate metrics for poor class (0)
            precision_poor = precision_score(self.y_test, self.y_pred, pos_label=0, zero_division=0) # type: ignore[arg-type]
            recall_poor = recall_score(self.y_test, self.y_pred, pos_label=0, zero_division=0) # type: ignore[arg-type] # = Spec(Good)
            f1_poor = f1_score(self.y_test, self.y_pred, pos_label=0, zero_division=0) # type: ignore[arg-type]

            logger.info(f"Metrics (Default Threshold ~0.5):")
            logger.info(f"  Overall: Acc={accuracy:.4f}")
            logger.info(f"  Good Outcome (1): Prc={precision_good:.4f}, Rec(TPR)={recall_good:.4f}, F1={f1_good:.4f}")
            logger.info(f"  Poor Outcome (0): Prc={precision_poor:.4f}, Rec(TPR)={recall_poor:.4f}, F1={f1_poor:.4f}")

        except Exception as e:
            logger.error(f"Error calculating classification metrics: {e}")

    def confusion_matrix(self):
        # Confusion Matrix at default threshold
        try:
            cm = confusion_matrix(self.y_test, self.y_pred)
            tn, fp, fn, tp = cm.ravel() # tn=TN(Good), fp=FP(Good), fn=FN(Good), tp=TP(Good)
            specificity_good = tn / (tn + fp) if (tn + fp) > 0 else 0 # TNR for class 1 (Good) = Spec(Good)
            recall_tpr_good = recall_score(self.y_test, self.y_pred, zero_division=0) # TPR for class 1 (Good) = Spec(Poor)
            logger.info(f"Confusion Matrix (Default Threshold ~0.5): [[TN(Good)={tn}, FP(Good)={fp}], [FN(Good)={fn}, TP(Good)={tp}]]")
            logger.info(f"  Metrics derived: Spec(Good)={specificity_good:.4f}, Spec(Poor)={recall_tpr_good:.4f}") # Explicit labels
            return cm
        except Exception as e:
            logger.error(f"Error calculating confusion matrix: {e}")
            return None

    def plot_confusion_matrix(self,thr):
        # Compute confusion matrix
        y_scores = self.model.predict_proba(self.X_test)[:, 1]
        y_pred = (y_scores >= thr).astype(int)
        cm = confusion_matrix(self.y_test, y_pred)

        labels = ["Good", "Poor"]

        # Plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap="Blues",
                    xticklabels=labels, yticklabels=labels, linewidths=0.5, square=True)
        
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Confusion Matrix with Threshold {thr:.2f}")

        cm_path = os.path.join(OUTPUT_PATH, f"cm_with_thr_{thr:.2f}.png")
        plt.savefig(cm_path, dpi=300, bbox_inches="tight")
        plt.show()


    def feature_importance(self):
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
             if OUTPUT_PATH:
                  save_path = os.path.join(OUTPUT_PATH, "feature_importance.csv")
                  perm_importance_df.to_csv(save_path, index=False)
                  logger.info(f"Feature importance saved: {save_path}")
             else:
                  logger.warning("Feature importance not saved (OUTPUT_PATH undefined/inaccessible).")
        except Exception as e:
             logger.error(f"Error calculating feature importance: {e}")


    def find_specificity(self, fpr, tpr, thresholds, target_fpr=None):
        best_fpr = -1.0
        best_tpr = -1.0
        best_idx = -1

        if target_fpr is not None:
            indices = np.where(fpr == target_fpr)[0]
            if len(indices) > 0:
                best_idx = indices[np.argmax(tpr[indices])]
                best_fpr = fpr[best_idx]
                best_tpr = tpr[best_idx]
                best_threshold = thresholds[best_idx]

        return best_threshold, best_fpr, best_tpr

    def find_recall(self, fpr, tpr, thresholds, target_tpr=None):
        best_fpr = -1.0
        best_tpr = -1.0
        best_idx = -1

        if target_tpr is not None:
            indices = np.where(tpr >= target_tpr)[0]
            if len(indices) > 0:
                best_idx = indices[np.argmin(fpr[indices])]
                best_fpr = fpr[best_idx]
                best_tpr = tpr[best_idx]
                best_threshold = thresholds[best_idx]

        return best_threshold, best_fpr, best_tpr


    def roc_curve_analysis(self):
        try:
            y_scores = self.model.predict_proba(self.X_test)[:, 1]
        except Exception as e:
            logger.error(f"Error getting model probabilities: {e}")
            return

        try:
            fpr, tpr, thresholds = roc_curve(self.y_test, y_scores) # fpr/tpr
            roc_auc = auc(fpr, tpr)
            logger.info(f"Overall ROC AUC: {roc_auc:.4f}")

            # --- Find Project Goal 1: Specificity(Poor) = 1.0, minimize FPR ---
            logger.info("-" * 20)
            logger.info("Analysis for Project Goal 1: Specificity = 1.0 or FPR = 0.0")
            thresh_poor, fpr_poor, tpr_poor = self.find_specificity(fpr, tpr, thresholds, target_fpr=0)

            # --- Find Project Goal 2: TPR >= 0.95 ---
            logger.info("-" * 20)
            logger.info("Analysis for Project Goal 2: Specificity(good) >= 0.95")
            thresh_good, fpr_good, tpr_good = self.find_recall(fpr, tpr, thresholds, target_tpr=0.95)

            # --- Plotting ---
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

            # Highlight Project Goal 1
            ax.scatter(fpr_poor, tpr_poor, s=100, facecolors='none', edgecolors='green', zorder=5, linewidth=1.5,
                    label=f'Goal 1: Spec(Poor)≈1 (Th~{thresh_poor:.2f})')

            # Highlight Project Goal 2
            ax.scatter(fpr_good, tpr_good, s=100, facecolors='none', edgecolors='blue', zorder=5, linewidth=1.5,
                    label=f'Goal 2: Spec(Good)≥.95 (Th~{thresh_good:.2f})')

            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ticks = np.arange(0.0, 1.1, 0.1)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve 24H')
            ax.legend(loc="lower right")
            ax.grid(True)

            # --- Save or Show Plot ---
            roc_save_path = os.path.join(OUTPUT_PATH, f"roc_curve_{ALL_FEATURES}_{FILES_USED}.png")
            fig.savefig(roc_save_path, dpi=150, bbox_inches='tight')
            logger.info(f"ROC curve plot saved: {roc_save_path}")
            plt.show()

            plt.close(fig)

        except Exception as e:
            logger.error(f"Error during ROC curve analysis: {e}")
        
        return thresh_poor,thresh_good
    
    def accuracy_in_thr(self,thr):
        y_scores = self.model.predict_proba(self.X_test)[:, 1]
        y_pred = (y_scores >= thr).astype(int)
        accuracy = accuracy_score(self.y_test, y_pred)
        logger.info(f"Accuracy at threshold {thr:.2f}: {accuracy:.4f}")
        return accuracy

