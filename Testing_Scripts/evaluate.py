import sys
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from Models.SmokeNet import SatelliteNet, predict
from Models.SmokeDataset import get_datasets

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

testing_data = get_datasets(False)

fname = sys.argv[1]
sd = torch.load(fname)
variant = sd["variant"]
model = SatelliteNet(variant)
model.load_state_dict(sd["model_state_dict"])
model.to(device)

y_pred = predict(model, testing_data)
y_pred = torch.stack(y_pred)
y_pred = torch.argmax(y_pred, axis=1).tolist()
y_true = [testing_data[i][1].item() for i in range(len(testing_data))]

print("The cohen kappa score: ", cohen_kappa_score(y_true, y_pred))
print("The f1_score score: ", f1_score(y_true, y_pred, average="macro"))
print("The accuracy: ", accuracy_score(y_true, y_pred))
print("The confusion matrix:")
print(confusion_matrix(y_true, y_pred))
