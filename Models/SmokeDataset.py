import torch
import torchvision.transofrms as tf

from collections import namedtuple
from math import floor
from torch import nn
from PIL import Image
from torch.utils.data import Dataset


class SmokeDataset(Dataset):
    """
    Extend PyTorch's Dataset for our specific purposes.
    """
    def __init__(self, path_df, image_ids, train=True):
        """
        Load the images corresponding to `image_ids` as PIL's Image objects and the corresponding labels as a torch tensor.
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
        self.transform = get_transform(train)
    

    def __getitem__(self, idx):
        return self.transform(self.X[idx]), self.y[idx]
    

    def __len__(self):
        return len(self.y)


def get_transforms(train=True):
    pil_to_tensor = tf.ToTensor()
    tensor_to_pil = tf.ToPILImage()
    crop = tf.RandomCrop((224, 224))
    hflip = tf.RandomHorizontalFlip()
    vflip = tf.RandomVerticalFlip()
    normalize = tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    if train:
        transforms = tf.Compose([crop, hflip, vflip, pil_to_tensor, normalize])
    else:
        transforms = tf.Compose([pil_to_tensor, normalize])
    return transforms
