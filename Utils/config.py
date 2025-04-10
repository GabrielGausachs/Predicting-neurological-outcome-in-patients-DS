import os
import torch

# -----------------------------------------
# Training configuration
# -----------------------------------------

MODEL = '2' # '1' for 12h model, '2' for 24h model
if MODEL == '1':
    FILES_USED = '1'
    MODEL_NAME = "RF_12h"
    SEED = 27738845
    RANDOM_FEATURE_SEED = 407654

elif MODEL == '2':
    FILES_USED = '2'
    MODEL_NAME = "RF_24h"
    SEED = 36034352
    RANDOM_FEATURE_SEED = 851720

RANDOM_SEED = None
TARGET_COLUMN = 'Patient Outcome'
PREPROCESS = False
TRAINING = False


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
