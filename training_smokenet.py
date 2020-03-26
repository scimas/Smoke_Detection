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


# Lets begin

image_path_df = pd.read_csv("Image_Paths.csv")

# encode the labels
label_encoding = {"Cloud": 0, "Dust": 1,
                  "Haze": 2, "Land": 3, "Seaside": 4, "Smoke": 5}
# replace the orignal labels with encoded labels
image_path_df.image_type.replace(label_encoding, inplace=True)


weight_denom = image_path_df.image_type.value_counts().values / 6
class_weights = len(image_path_df) / weight_denom


# 20% of data is testing data, get its indices
train_valid_indices, test_indices = train_test_split(np.arange(len(image_path_df)), 
test_size=0.2, stratify=image_path_df.image_type)
# 16% of the original data is validation data, which comes out to be 20% of combined training and validation data
# Again, get its and training set's indices
train_indices, valid_indices = train_test_split(train_valid_indices, test_size=0.2, stratify=image_path_df.image_type[train_valid_indices])

train_data = SmokeDataset(image_path_df, train_indices, train=True)
# train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())

validation_data = SmokeDataset(image_path_df, valid_indices, train=False)
# validation_loader = DataLoader(train_data, batch_size=256, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())

test_data = SmokeDataset(image_path_df, test_indices, train=False)
# test_loader = DataLoader(train_data, batch_size=256, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())
