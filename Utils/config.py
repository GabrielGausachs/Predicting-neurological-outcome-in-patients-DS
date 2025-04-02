import os
import torch

# -----------------------------------------
# Training configuration
# -----------------------------------------

FILES_USED = '1' # '1' for 12h, '2' for 24h, '3' for both files
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

N_ESTIMATORS = [50, 100, 200, 300]
MAX_FEATURES = ['sqrt', 'log2', 0.3]
MAX_DEPTH = [None, 10, 20, 30]
JOBS = -1 # Number of jobs to run in parallel for RandomForestClassifier
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#DEVICE = torch.device("cpu")
RANDOM_SEED = 23