import os
import torch

# -----------------------------------------
# Training configuration
# -----------------------------------------

FILES_USED = '1' # '1' for 12h, '2' for 24h, '3' for both files
TARGET_COLUMN = 'Patient Outcome'
PREPROCESS = False
ALL_FEATURES = True # If False, only keep features in FEATURES_TO_KEEP
#FEATURES_TO_KEEP_12 = ['BSR','abs(renyi)','fhtife2','beta_tot']
#FEATURES_TO_KEEP_24 = ['SkewAM','abs(shan)','KurtAM','skewness','beta_theta','BSR','spindle_theta','meanAM','fhtife3','alpha','beta_tot','Complexity','spindle','alpha_delta']
FEATURES_TO_KEEP = None
MODEL_NAME = "RF_12h"
TRAINING = True


# -----------------------------------------
# Paths
# -----------------------------------------

LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Logs")
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Data")
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Outputs")
MODELS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Models")

# -----------------------------------------
# Parameters
# -----------------------------------------

N_ESTIMATORS = [100,150,200]
MAX_FEATURES = [6,8,10]
MAX_DEPTH = [None]
JOBS = -1 # Number of jobs to run in parallel for RandomForestClassifier
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#DEVICE = torch.device("cpu")
RANDOM_SEED = None
