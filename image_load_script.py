# Basic packages 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random



import torch
import torch.nn as nn

from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn import metrics
from PIL import Image
from keras.preprocessing.image import  load_img, img_to_array

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Lets begin

image_path_df = pd.read_csv("Image_Paths.csv")
print(image_path_df.head())

image_arrays = []
for p in image_path_df.image_path:
    im = load_img(p, color_mode="rgb",target_size=(256,256), interpolation="nearest")
    img_array = img_to_array(im).reshape(3,256,256)
    #reshape array here
    image_arrays.append(img_array)

print(image_arrays[0].shape)


image_path_df.image_type.replace({"Cloud":0, "Dust": 1, "Haze": 2, "Land": 3, "Seaside": 4, "Smoke":5}, inplace=True)
# Here I have calssified them as separate classes, but we have to find a way to work with multi-labeling.
# Refer Dataset -> Classes section of the paper

Y = image_path_df.image_type[:500].values
X = np.array(image_arrays[:500])

print("After converting to arrays: ", X.shape)



x_train, x_test, y_train, y_test = train_test_split(X,Y, stratify = Y, test_size = 0.2, random_state= 42)
x_train, x_validate, y_train, y_validate = train_test_split(x_train,y_train, stratify=y_train, test_size=0.1, random_state=40)


x_train = torch.from_numpy(x_train).to(device)
x_test = torch.from_numpy(x_test).to(device)
y_train = torch.from_numpy(y_train).to(device)
y_test = torch.from_numpy(y_test).to(device)
x_validate = torch.from_numpy(x_validate).to(device)
y_validate = torch.from_numpy(y_validate).to(device)


# x_train = torch.from_numpy(x_train)
# x_test = torch.from_numpy(x_test)
# y_train = torch.from_numpy(y_train)
# y_test = torch.from_numpy(y_test)
# x_validate = torch.from_numpy(x_validate)
# y_validate = torch.from_numpy(y_validate)



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


print("Starting training loop...")
for epoch in range(N_EPOCHS):

    loss_train = 0
    model.train()
    # m = nn.Sigmoid()
    min_loss = 2
    for batch in range(len(x_train)//BATCH_SIZE + 1):
        inds = slice(batch*BATCH_SIZE, (batch+1)*BATCH_SIZE)
        optimizer.zero_grad()
        logits = model(x_train[inds])

        loss = criterion(logits,y_train[inds].long())
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
    if(min_loss > (loss_train/BATCH_SIZE)):
        print("=> Saving a new best")
        device = torch.device("cpu")
        model.to(device)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'learn_rate': LR,
            "DROPOUT": DROPOUT
        }, "model_smoke_detect.pt")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

    else:
        print("=> Training loss did not improve")

    model.eval()
    print("Epoch {} | Train Loss {:.5f}".format(
         epoch, loss_train / BATCH_SIZE))

# print("----------------Testing the model-------------------------")
# device = torch.device("cpu")
# checkpoint = torch.load("model_smoke_detect.pt")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = CNN(checkpoint['DROPOUT']).to(device)
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer = torch.optim.SGD(model.parameters(), lr = checkpoint['learn_rate'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#
# model.eval()
# with torch.no_grad():
#     y_test_pred = model(x_test.to(device))
#     loss = criterion(y_test_pred, y_test.to(device))
#     loss_test = loss.item()
# print("Test Loss {:.5f}".format(loss_test))
