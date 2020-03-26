# Basic packages 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn import metrics
from Models.SmokeDataset import SmokeDataset
from torch.utils.data import DataLoader

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
device = nn.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Lets begin

image_path_df = pd.read_csv("Image_Paths.csv")
print(image_path_df.head())
label_encoding = {"Cloud":0, "Dust": 1, "Haze": 2, "Land": 3, "Seaside": 4, "Smoke":5}
image_path_df.image_type.replace(label_encoding, inplace=True)

class_weights = len(image_path_df) / image_path_df.image_type.value_counts().values / 6
# 20% of data is testing data, get its indices
train_valid_indices, test_indices = train_test_split(np.arange(len(image_path_df)), test_size=0.2, stratify=image_path_df.image_type)

test_data = SmokeDataset(image_path_df, test_indices, train=False)
test_loader = DataLoader(train_data, batch_size=256, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())



print("----------------Testing the model-------------------------")
device = torch.device("cpu")
checkpoint = torch.load("model_smoke_detect.pt")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CNN(checkpoint['DROPOUT']).to(device)
model.load_state_dict(checkpoint['model_state_dict'])

with torch.no_grad():
        test_loss = 0
        model.eval()
        for i, (images, labels) in enumerate(test_loader):
            logits = model(images.to(device))
            test_loss += criterion(logits,labels.to(device)).item() * len(labels)
        # Average validation loss.
        test_loss /= len(test_loader)
print("Test Loss {:.5f}".format(test_loss))
