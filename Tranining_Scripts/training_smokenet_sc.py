# Basic packages

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import os
# import random
import torch
import torch.nn as nn

from sklearn.metrics import cohen_kappa_score, f1_score
# from sklearn import metrics
from Models.SmokeNet import SmokeNet, fit, predict
from datetime import datetime
from Models.SmokeDataset import get_datasets

# Set-up

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class_weights, training_data, validation_data, testing_data = get_datasets()

learn_rate = 0.01
n_epochs = 200
batch_size = 32
variant = "SC"
smoke = SmokeNet(sc_cs=variant)
smoke.to(device)
if torch.cuda.device_count() > 1:
    print("Using multiple GPUs")
    smoke = nn.DataParallel(smoke)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
optimizer = torch.optim.SGD(smoke.parameters(), lr=learn_rate, momentum=0.9)

filename = "Smokenet_trainlog_"+variant.upper()+"_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
fit(smoke, optimizer, criterion, training_data, validation_data, class_weights, n_epochs, batch_size, filename)
y_pred = predict(smoke, testing_data)
y_pred = torch.cat(y_pred)
y_pred = torch.argmax(y_pred, axis=1).tolist()
y_true = testing_data[:][1].tolist()


print("The cohen kappa score is:", cohen_kappa_score(y_true, y_pred))
print("The f1_score score is:", f1_score(y_true, y_pred, average="macro"))
