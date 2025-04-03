from sklearn.metrics import (confusion_matrix, accuracy_score, 
                             precision_score, recall_score, f1_score)
from sklearn.inspection import permutation_importance
from Utils.logger import initialize_logger, get_logger
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

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
        self.y_pred = y_pred

    def metrics(self):
        # Generate metrics
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred)
        recall = recall_score(self.y_test, self.y_pred)
        f1 = f1_score(self.y_test, self.y_pred)
        logger.info("Metrics generated")
        logger.info(f"Accuracy: {accuracy}")
        logger.info(f"Precision: {precision}")
        logger.info(f"Recall: {recall}")
        logger.info(f"F1 Score: {f1}")
        logger.info("-" * 50)
    
    def confusion_matrix(self):
        # Generate confusion matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        logger.info("Confusion matrix generated")
        logger.info(cm)
        logger.info("-" * 50)
        return cm
    
    def feature_importance(self):
        # Generate feature importance
        result = permutation_importance(self.model, self.X_test, self.y_test, n_repeats=10, random_state=RANDOM_SEED, n_jobs=JOBS)

        importances_perm_mean = result.importances_mean
        importances_perm_std = result.importances_std

        feature_names = self.X_test.columns.tolist()

        perm_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance_Perm_Mean': importances_perm_mean,
            'Importance_Perm_Std': importances_perm_std
        })
        perm_importance_df = perm_importance_df.sort_values(by='Importance_Perm_Mean', ascending=False)
        perm_importance_df.to_csv(f"{OUTPUT_PATH}/feature_importance.csv", index=False)
        logger.info("Feature importance generated")
        logger.info("-" * 50)

    def roc_curve(self):
        fpr, tpr, thresholds = roc_curve(self.y_test, self.model.predict_proba(self.X_test)[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        #plt.savefig(f"{OUTPUT_PATH}/roc_curve.png")
        plt.show()
        plt.close()
        logger.info("ROC curve generated")

        print(f"False Positive Rate: {fpr}")
        print(f"True Positive Rate: {tpr}")
        print(f"Thresholds: {thresholds}")