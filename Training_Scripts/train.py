import sys
import torch
from torch import nn
from Models.SmokeNet import SatelliteNet, fit
from Models.SmokeDataset import get_datasets
from datetime import datetime

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class_weights, training_data, validation_data = get_datasets(True)

if len(sys.argv) > 2:
    variant = sys.argv[1]
    model_suffix = variant + "_" + sys.argv[2]
elif len(sys.argv) > 1:
    variant = sys.argv[1]
    model_suffix = variant
else:
    variant = "sc"
    model_suffix = variant + "_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

learn_rate = 0.0001
decay = 1.0
n_epochs = 200
batch_size = 16
model = SatelliteNet(sc_cs=variant)
model.to(device)

if torch.cuda.device_count() > 1:
    print("Using multiple GPUs")
    model = nn.DataParallel(model)
    batch_size *= torch.cuda.device_count()

criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
optimizer = torch.optim.AdamW(model.parameters(), lr=learn_rate, weight_decay=decay)

fit(model, optimizer, criterion, training_data, validation_data, class_weights, n_epochs, batch_size, model_suffix)
