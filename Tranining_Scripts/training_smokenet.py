# Basic packages

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import os
# import random
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, f1_score
# from sklearn import metrics
from Models.SmokeDataset import SmokeDataset
from Models.SmokeNet import SmokeNet, predict
from torch.utils.data import DataLoader

from Models.SmokeDataset import get_datasets

# Set-up

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



class_weights, training_data, validation_data, testing_data = get_datasets()

smoke = SmokeNet()
smoke.fit(training_data, validation_data)
y_pred = predict(testing_data)
y_true = testing_data[1]


print("The cohen kappa score is:", cohen_kappa_score(y_true, y_pred))
print("The f1_score score is:", f1_score(y_true, y_pred))

