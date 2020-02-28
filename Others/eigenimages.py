import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA


def read_image_meta_df():
    return pd.read_csv("Image_Paths.csv", header=0)


def create_pca(image_class: str, image_meta_df: pd.DataFrame) -> PCA:
    image_paths = image_meta_df.loc[
        image_meta_df["image_type"] == image_class, "image_path"]
    image_df = pd.DataFrame()
    for path in image_paths:
        im = Image.open(path)
        im = im.convert(mode="L")
        im = im.resize((128, 128), resample=Image.BICUBIC)
        im = np.asarray(im).flatten()
        im = pd.Series(im, name=os.path.basename(path))
        image_df = image_df.append(im)
    
    img_pca = PCA(n_components=0.8)
    img_pca.fit(image_df)
    return img_pca


def plot_pca_components(image_pca: PCA, image_class: str):
    n_components = image_pca.n_components_
    n_row = n_col = math.ceil(math.sqrt(n_components))
    eig_images = image_pca.components_.reshape((n_components, 128, 128))
    eig_images = np.asarray(eig_images)
    fig, axes = plt.subplots(n_row, n_col)
    for img, ax in zip(eig_images, axes.flat):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(img, cmap="gray")
    for i in range(n_components, n_row*n_col):
        axes.flat[i].set_xticks([])
        axes.flat[i].set_yticks([])
    fig.suptitle(image_class + " Eigenimages\nVariance Explained = 0.8")
    plt.savefig(image_class + "_eigen.png")
    plt.show()
