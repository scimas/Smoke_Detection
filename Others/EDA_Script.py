import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

image_types = [p for p in os.listdir("data/") if (not p.startswith("."))]
print("The type of images are:", image_types)

imageList_df = pd.DataFrame(columns=["image_path","image_type"])
img_types = []
img_counts = []
row = 0
for im in image_types:
    image_list = os.listdir("data/"+im+"/")
    img_types.append(im)
    img_counts.append(len(image_list))
    print("No of images in {0} type are {1}".format(im,len(image_list)))
    im_paths = ["data/"+im+"/"+i for i in image_list]
    temp_df = pd.DataFrame({"image_path":im_paths, "image_type":[im]*len(image_list)})
    imageList_df = pd.concat([imageList_df,temp_df])

print("Total Number of images:", len(imageList_df))
print(imageList_df.head())

# write to csv
imageList_df.to_csv("Image_Paths.csv",index=False)


# Plot the counts in barplot
plt.figure(figsize=(5,5))
plt.bar(img_types,img_counts,width= 0.8, color=['black', 'red', 'magenta', 'blue', 'cyan', 'green'])
plt.title("Types of Images VS Count")
plt.xlabel("Image Type")
plt.ylabel("No of images")
plt.show()


# Read the images
img_cloud = mpimg.imread('data/Cloud/cloud_1.tif')
img_dust = mpimg.imread('data/Dust/dust_1.tif')
img_haze = mpimg.imread('data/Haze/haze_1.tif')
img_land = mpimg.imread('data/Land/land_1.tif')
img_seaside = mpimg.imread('data/Seaside/seaside_1.tif')
img_smoke = mpimg.imread('data/Smoke/smoke_10.tif')


# Plot the sample images

fig, axs = plt.subplots(3,2, figsize=(5,5))
axs[0, 0].imshow(img_cloud)
axs[0, 0].set_title('Cloud')

axs[0, 1].imshow(img_dust)
axs[0, 1].set_title('Dust')

axs[1, 0].imshow(img_haze)
axs[1, 0].set_title('Haze')

axs[1, 1].imshow(img_land)
axs[1, 1].set_title('Land')

axs[2, 0].imshow(img_seaside)
axs[2, 0].set_title("Seaside")

axs[2, 1].imshow(img_smoke)
axs[2, 1].set_title("Smoke")

plt.tight_layout(pad=3.0)

plt.show()
