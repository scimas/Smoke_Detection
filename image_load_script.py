# Basic packages 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random


from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn import metrics
from PIL import Image
from keras.preprocessing.image import  load_img, img_to_array



# Lets begin

image_path_df = pd.read_csv("Image_paths.csv")
print(image_path_df.head())

image_arrays = []
for p in image_path_df.image_path:
    im = load_img(p, color_mode="rgb",target_size=(256,256), interpolation="nearest")
    img_array = img_to_array(im)
    image_arrays.append(img_array)

print(image_arrays[0].shape)


image_path_df.image_type.replace({"Cloud":0, "Dust": 1, "Haze": 2, "Land": 3, "Seaside": 4, "Smoke":5}, inplace=True)
# Here I have calssified them as separate classes, but we have to find a way to work with multi-labeling.
# Refer Dataset -> Classes section of the paper

Y = to_categorical(image_path_df.image_type)
X = np.array(image_arrays)



x_train, x_test, y_train, y_test = train_test_split(X,Y, stratify = Y, test_size = 0.2, random_state= 42)
x_train, x_validate, y_train, y_validate = train_test_split(x_train,y_train, stratify=y_train, test_size=0.1, random_state=40)
