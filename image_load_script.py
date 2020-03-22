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
from Models.modelling import SmokeDataset
from torch.utils.data import DataLoader

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
# 16% of the original data is validation data, which comes out to be 20% of combined training and validation data
# Again, get its and training set's indices
train_indices, valid_indices = train_test_split(train_valid_indices, test_size=0.2, stratify=image_path_df.image_type[train_valid_indices])

train_data = SmokeDataset(image_path_df, train_indices, train=True)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())

validation_data = SmokeDataset(image_path_df, valid_indices, train=False)
validation_loader = DataLoader(train_data, batch_size=256, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())

test_data = SmokeDataset(image_path_df, test_indices, train=False)
test_loader = DataLoader(train_data, batch_size=256, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())

print("######The device is: ", device)
LR = 0.05
N_EPOCHS = 20
BATCH_SIZE = 50
DROPOUT = 0.5

# %% -------------------------------------- CNN Class ------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self, DROPOUT):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(5, 5)) # 252 x 252 x 32
        self.convnorm1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(3) # 84 x 84 x 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size= (3, 3)) # 82 x 82 x 64
        self.convnorm2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(3) # 27 x 27 x 64
        self.linear1 = nn.Linear(64*27*27, 400)
        self.linear1_bn = nn.BatchNorm1d(400)
        self.drop = nn.Dropout(DROPOUT)
        self.linear2 = nn.Linear( 400, 6)
        self.act = torch.relu
        self.flat = nn.Flatten()

    def forward(self, x):
        x = self.pool1(self.convnorm1(self.act(self.conv1(x))))
        x = self.pool2(self.convnorm2(self.act(self.conv2(x))))
        x = self.drop(self.linear1_bn(self.act(self.linear1(self.flat(x)))))
        return self.linear2(x)

model = CNN(DROPOUT).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()
min_validation_loss = 1e10

print("Starting training loop...")
for epoch in range(N_EPOCHS):

    loss_train = 0
    model.train()
    min_loss = 2
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        logits = model(images.to(device))
        loss = criterion(logits,labels.to(device))
        loss.backward()
        optimizer.step()
        loss_train += loss.item() * len(labels)
    # Average training loss. Less meaningful since model is being updated on each minibatch.
    loss_train /= len(train_loader)
    with torch.no_grad():
        validation_loss = 0
        model.eval()
        for i, (images, labels) in enumerate(validation_loader):
            logits = model(images.to(device))
            validation_loss += criterion(logits,labels.to(device)).item() * len(labels)
        # Average validation loss.
        validation_loss /= len(validation_loader)
        if validation_loss < min_validation_loss:
            min_validation_loss = validation_loss
            print("=> Saving a new best")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'learn_rate': LR,
                "DROPOUT": DROPOUT
            }, "model_smoke_detect.pt")
        else:
            print("=> Training loss did not improve")
        print("Epoch {} | Validation Loss {:.5f}".format(epoch, validation_loss))
optimizer.zero_grad()

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
