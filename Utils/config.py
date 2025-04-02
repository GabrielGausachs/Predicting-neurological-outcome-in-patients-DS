import os
import torch

# -----------------------------------------
# Training configuration
# -----------------------------------------

MODELNAME = 'resnet18'
CRITERION = 'CrossEntropyLoss'
OPTIMIZER = "Adam"
DEFENSE_MODEL = "DUNet"
FILES_USED = '1' # '1' for 12h, '2' for 24h, '3' for both files
TARGET_COLUMN = 'Patient Outcome'


# -----------------------------------------
# Main steps
# -----------------------------------------


# -----------------------------------------
# Paths 
# -----------------------------------------

LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Logs")
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Data")

# -----------------------------------------
# Parameters 
# -----------------------------------------

BATCH_SIZE_ATTACK = 16
BATCH_SIZE_UNET = 32
NUM_WORKERS = 0
IMAGES_TO_TEST = 15000
EPSILON = 0.03
STEPSIZE = 0.005
NUM_ITERATIONS = 10
LEARNING_RATE = 0.01
EPOCHS = 20


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#DEVICE = torch.device("cpu")
RANDOM_SEED = 23