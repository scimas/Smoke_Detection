import torch
import torchvision.transforms as tf
import pandas as pd
import numpy as np
from collections import namedtuple
from math import floor
from torch import nn
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class SmokeDataset(Dataset):
    """
    Extend PyTorch's Dataset for our specific purposes.
    """
    def __init__(self, path_df = pd.read_csv("Image_Paths.csv")):
        """
        Load the images corresponding to `image_ids` as PIL's Image objects and the corresponding labels as a torch tensor.
        """

        # DEfault image and label dataset
        self.image_path_df = path_df

        # encode the labels
        label_encoding = {"Cloud": 0, "Dust": 1, "Haze": 2, "Land": 3, "Seaside": 4, "Smoke": 5}

        weight_denom = self.image_path_df.image_type.value_counts().values / 6

        class_weights = len(self.image_path_df) / weight_denom

        # 20% of data is testing data, get its indices

        train_valid_indices, test_indices = train_test_split(np.arange(len(self.image_path_df)), 
        test_size=0.2, stratify=self.image_path_df.image_type)
        # 16% of the original data is validation data, which comes out to be 20% of combined training and validation data

        # Again, get its and training set's indices

        train_indices, valid_indices = train_test_split(train_valid_indices, test_size=0.2, stratify=self.image_path_df.image_type[train_valid_indices])

        self.train_data = self.get_data(train_indices, train=True)
        # train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())

        self.validation_data = self.get_data(valid_indices, train=False)
        # validation_loader = DataLoader(train_data, batch_size=256, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())

        self.test_data = self.get_data(test_indices, train=False)
        # test_loader = DataLoader(train_data, batch_size=256, shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available())

        



    def get_data(self, image_ids=[], train=True):

        if len(image_ids) == 0:
            image_ids = np.arange(len(self.image_path_df))


        path_df =self.image_path_df
        if train:
            downsize = 230
        else:
            downsize = 224
        X = [
            Image.open(path).resize((downsize, downsize), resample=Image.LANCZOS)
            for path in path_df.loc[image_ids, "image_path"]
        ]
        y = torch.from_numpy(path_df.loc[image_ids, "image_type"].values)
        self.transform = self.get_transforms(train)
        
        return X,y
    

    def __getitem__(self, idx):
        # return self.transform(self.X[idx]), self.y[idx]
        return self.transform(self.get_data([idx]))
    

    def __len__(self):
        return len(self.image_path_df)


    def get_transforms(self,train=True):
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
