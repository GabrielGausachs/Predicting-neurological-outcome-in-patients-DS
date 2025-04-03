import os
import torch

# -----------------------------------------
# Training configuration
# -----------------------------------------

FILES_USED = '2' # '1' for 12h, '2' for 24h, '3' for both files
TARGET_COLUMN = 'Patient Outcome'
PREPROCESS = False


# -----------------------------------------
# Main steps
# -----------------------------------------


# -----------------------------------------
# Paths
# -----------------------------------------

LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Logs")
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Data")
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Outputs")

# -----------------------------------------
# Parameters
# -----------------------------------------

N_ESTIMATORS = [100,150,200,250]
MAX_FEATURES = [6,8,10,12]
MAX_DEPTH = [None,30,50]
JOBS = 10 # Number of jobs to run in parallel for RandomForestClassifier
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#DEVICE = torch.device("cpu")
RANDOM_SEED = None
