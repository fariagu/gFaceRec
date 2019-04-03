from __future__ import absolute_import, division, print_function

from skimage.io import imread
from skimage.transform import resize
import numpy as np

import random as rand

import keras
from keras.preprocessing.image import ImageDataGenerator

import utils

def get_transform():
    range_shift = int(5 * 160 / 100)

    theta = rand.randint(-15, 15)
    tx = rand.randint(-range_shift, range_shift)
    ty = rand.randint(-range_shift, range_shift)
    flip = rand.randint(0, 1)

    transform = {
        "theta": theta,
        "tx": tx,
        "ty": ty,
        "flip_horizontal": True if flip == 0 else False
    }

    return transform

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

            image_array.append(
                resize(
                    im_gen.apply_transform(
                        x=imread(file_name),
                        transform_parameters=get_transform()
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