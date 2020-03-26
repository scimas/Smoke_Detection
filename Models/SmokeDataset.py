import torch
import numpy as np
import pandas as pd

from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms as tf
from torch.utils.data import Dataset


class SmokeDataset(Dataset):
    """
    Extend PyTorch's Dataset for our specific purposes.
    """
    def __init__(self, path_df: pd.DataFrame, image_ids: list, train=True):
        """
        Load the images corresponding to `image_ids` as PIL's Image objects and the corresponding labels as a torch tensor.
        Image paths are read from the `path_df` `DataFrame`.
        Value of `train` determined which transforms to apply to the images.
        """
        if train:
            downsize = 230
        else:
            downsize = 224
        self.X = [
            Image.open(path).resize((downsize, downsize), resample=Image.LANCZOS)
            for path in path_df.loc[image_ids, "image_path"]
        ]
        self.y = torch.from_numpy(path_df.loc[image_ids, "image_type"].values)
        self.transform = get_transforms(train)

    def __getitem__(self, idx):
        return self.transform(self.X[idx]), self.y[idx]

    def __len__(self):
        return len(self.y)


def get_transforms(train=True):
    pil_to_tensor = tf.ToTensor()
    crop = tf.RandomCrop((224, 224))
    hflip = tf.RandomHorizontalFlip()
    vflip = tf.RandomVerticalFlip()
    normalize = tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    if train:
        transforms = tf.Compose([crop, hflip, vflip, pil_to_tensor, normalize])
    else:
        transforms = tf.Compose([pil_to_tensor, normalize])
    return transforms


def get_datasets():
    image_path_df = pd.read_csv("Image_Paths.csv")
    label_encoding = {"Cloud": 0, "Dust": 1, "Haze": 2, "Land": 3, "Seaside": 4, "Smoke": 5}
    image_path_df.image_type.replace(label_encoding, inplace=True)

    class_weights = len(image_path_df) / image_path_df.image_type.value_counts().values / 6
    # 20% of data is testing data, get its indices
    train_valid_indices, test_indices = train_test_split(
        np.arange(len(image_path_df)), test_size=0.2,
        stratify=image_path_df.image_type
    )
    # 16% of the original data is validation data, which comes out to be 20% of combined training and validation data
    # Again, get its and training set's indices
    train_indices, valid_indices = train_test_split(
        train_valid_indices, test_size=0.2,
        stratify=image_path_df.image_type[train_valid_indices]
    )

    train_data = SmokeDataset(image_path_df, train_indices, train=True)
    validation_data = SmokeDataset(image_path_df, valid_indices, train=False)
    test_data = SmokeDataset(image_path_df, test_indices, train=False)

    return class_weights, train_data, validation_data, test_data
