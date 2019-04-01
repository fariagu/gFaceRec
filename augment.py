from __future__ import absolute_import, division, print_function

import os
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from matplotlib import pyplot

img = np.array(resize(imread("C:/datasets/CelebA/test_split/1/027827.jpg"), (224, 224)))

# datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
# datagen = ImageDataGenerator(rotation_range=90)
# datagen = ImageDataGenerator(shear_range=10.0)
# datagen = ImageDataGenerator(width_shift_range=0.1)
# datagen = ImageDataGenerator(height_shift_range=0.1)
datagen = ImageDataGenerator(
        rotation_range=2,
        shear_range=5,
        width_shift_range=2,
        height_shift_range=2,
    )

datagen.fit([img])


for resx in datagen.flow(np.array([img])):
    pyplot.imshow(resx[0])
    pyplot.show()
    break
# print(array.shape)

# pyplot.imshow(img)
# pyplot.show()