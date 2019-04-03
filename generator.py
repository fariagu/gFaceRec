from __future__ import absolute_import, division, print_function

from skimage.io import imread
from skimage.transform import resize
import numpy as np

import random as rand

import keras
from keras.preprocessing.image import ImageDataGenerator

import utils

class Generator(keras.utils.Sequence):

    def __init__(self, image_filenames, labels, batch_size):
        self.images, self.labels = image_filenames, labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.labels) / float(self.batch_size)))

    def __getitem__(self, idx):
        image_batch = self.images[idx * self.batch_size:(idx + 1) * self.batch_size]
        label_batch = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        transform = {
            "flip_horizontal": True,
            "brightness": 0.01
        }

        im_gen = ImageDataGenerator()

        image_array = []
        for file_name in image_batch:
            seed = rand.randint(1, 4)
            if seed == 1:
                transform = {
                    "flip_horizontal": True,
                    "brightness": 0.01
                }
            elif seed == 2:
                transform = {
                    "flip_horizontal": False,
                    "brightness": 0.01
                }
            elif seed == 3:
                transform = {
                    "flip_horizontal": False,
                }
            else:
                transform = {
                    "flip_horizontal": True,
                }

            image_array.append(
                resize(
                    im_gen.apply_transform(
                        x=imread(file_name),
                        transform_parameters=transform
                    ),
                    (utils.image_width, utils.image_width)
                )
            )
        
        return np.array(image_array), np.array(label_batch)

        # return np.array([
        #     resize(imread(file_name), (utils.image_width, utils.image_width))
        #        for file_name in image_batch]
        # ),
        # np.array(label_batch)