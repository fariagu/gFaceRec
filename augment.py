from __future__ import absolute_import, division, print_function

import os
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator

import random as r

from matplotlib import pyplot

import utils

img = np.array(resize(imread("C:/Users/gustavo.faria/Downloads/067807.jpg"), (160, 160)))

range_patch = int(160 / 3)
x_patch = r.randint(0, utils.image_width - range_patch)
y_patch = r.randint(0, utils.image_width - range_patch)

img = cv2.rectangle(img, (x_patch, y_patch), (x_patch + range_patch, y_patch + range_patch), (240, 234, 214), -1)

datagen = ImageDataGenerator(
        # rotation_range=2,
        # shear_range=5,
        # width_shift_range=2,
        # height_shift_range=2,
        # horizontal_flip=True,
    )
# 5  ---- 100
# x  ---- image_width

range_shift = 5 * 160 / 100

tx_seed = r.uniform(-range_shift, range_shift)
img = datagen.apply_transform(img, {"tx": tx_seed})

datagen.fit([img])


for resx in datagen.flow(np.array([img])):
    pyplot.imshow(resx[0])
    pyplot.show()
    break
# print(array.shape)

# pyplot.imshow(img)
# pyplot.show()