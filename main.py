import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
import os 
import pandas as pd
import json
from datetime import datetime

from Utils import (
    logger,
    dataloader
)

from Utils.config import (
    DATA_PATH,
    RANDOM_SEED,
    FILES_USED,
    TARGET_COLUMN,
    DEVICE,
)


if __name__ == "__main__":

    # Initialize logger
    logger.initialize_logger()
    logger = logger.get_logger()

    logger.info("-" * 50)
    logger.info("Executing main")
    logger.info("-" * 50)

    # Load data
    data_loader = dataloader.DataLoader()
    data_loader.load_data()
    data_loader.preprocess_data()
