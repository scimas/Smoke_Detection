import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img_cloud = mpimg.imread('data/Cloud/cloud_1.tif')
img_dust = mpimg.imread('data/Dust/dust_1.tif')
img_haze = mpimg.imread('data/Haze/haze_1.tif')
img_land = mpimg.imread('data/Land/land_1.tif')
img_seaside = mpimg.imread('data/Seaside/seaside_1.tif')
img_smoke = mpimg.imread('data/Smoke/smoke_1.tif')
# print(img)
# plt.imshow(img_cloud)
# plt.show()
# plt.imshow(img_dust)
# plt.show()
# plt.imshow(img_haze)
# plt.show()
# plt.imshow(img_land)
# plt.show()
# plt.imshow(img_seaside)
# plt.show()
# plt.imshow(img_smoke)
# plt.show()



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

plt.show()