# Smoke Detection
Deep Learning model to identify smoke from satellite images 

## Usage
Example training and testing scripts are included in the `Training_Scripts` and
`Testing_Scripts` directories.

### Training
From within the repository:

```
python3 -m Training_Scripts.train <variant> <suffix>
```

Where `<variant>` is one of `SC`, `CS`, `S`, `C` for spatial-channel,
channel-spatial, spatial and channel attention respectively. The `<suffix>` is
any string to distinguish the model save file from other files. This will create
a model file called `model_<variant>_<suffix>.pt`, a training log file and a
`CSV` file of the training and validation loss with epochs.

### Testing
From within the repository:

```
python3 -m Testing_Scripts.evaluate <filename>
```

`<filename>` is the saved model file's name. The model variant will be taken
from the file.It will print out Cohen's Kappa, F-1 score (macro averaged),
accuracy score and the confusion matrix.
