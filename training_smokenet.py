# Basic packages

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import os
# import random
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
# from sklearn import metrics
from Models.SmokeDataset import SmokeDataset
from torch.utils.data import DataLoader

# Set-up

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


dataset_instance = SmokeDataset()
