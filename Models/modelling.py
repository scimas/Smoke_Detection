import torch

from collections import namedtuple
from math import floor
from torch import nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

ConvSize = namedtuple(
    "ConvSize",
    ("out_channels", "kernel_size", "padding", "pool", "pool_padding"),
    defaults=(1, 3, 0, 2, 0)
)


class Classifier(nn.Module):
    def __init__(self, conv_sizes, fc_sizes):
        super(Classifier, self).__init__()
        in_size = 256
        out_size = 0
        conv_sizes.insert(0, ConvSize())
        self.layers = nn.ModuleDict()
        for i, layer in enumerate(conv_sizes[1:]):
            self.layers["conv" + str(i+1)] = nn.ModuleList(
                [
                    nn.Conv2d(conv_sizes[i][0], layer.out_channels, layer.kernel_size, padding=layer.padding),
                    # nn.Dropout2d(0.4),
                    nn.ReLU(),
                    nn.MaxPool2d(layer.pool, padding=layer.pool_padding)
                ]
            )
            mid_size = in_size + 2 * layer.padding - layer.kernel_size + 1
            out_size = floor((mid_size + 2 * layer.pool_padding - layer.pool) / layer.pool + 1)
            in_size = out_size
        
        self.layers["fc1"] = nn.ModuleList(
            [
                nn.Flatten(),
                nn.Linear(out_size * out_size * conv_sizes[-1].out_channels, fc_sizes[0]),
                nn.ReLU(),
            ]
        )
        for i, layer in enumerate(fc_sizes[1:]):
            if i == len(fc_sizes) - 1:
                self.layers["fc" + str(i+2)] = nn.ModuleList(
                    [
                        nn.Linear(fc_sizes[i], layer)
                    ]
                )
            else:
                self.layers["fc" + str(i+2)] = nn.ModuleList(
                    [
                        nn.Linear(fc_sizes[i], layer),
                        nn.ReLU()
                    ]
                )


    def forward(self, X):
        for name, layerlist in self.layers.items():
            for layer in layerlist:
                X = layer(X)

        return X


class SmokeDataset(Dataset):
    """
    Extend PyTorch's Dataset for our specific purposes.
    """
    def __init__(self, path_df, image_ids, transform=None):
        """
        Load the images corresponding to `image_ids` as PIL's Image objects and the corresponding labels as a torch tensor.
        """
        self.X = [
            Image.open(path).resize((230, 230), resample=Image.LANCZOS)
            for path in path_df.loc[image_ids, "image_path"]
        ]
        self.y = torch.from_numpy(path_df.loc[image_ids, "image_type"].values)
        if transform:
            self.transform = transform
        else:
            self.transform = ToTensor()
    

    def __getitem__(self, idx):
        return self.transform(self.X[idx]), self.y[idx]
    

    def __len__(self):
        return len(self.y)
