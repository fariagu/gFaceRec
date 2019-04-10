from __future__ import absolute_import, division, print_function

from skimage.io import imread
from skimage.transform import resize
import numpy as np

import random as rand

import keras

import utils

class Binary_Generator(keras.utils.Sequence):

    def __init__(self, image_filenames, labels, batch_size):
        self.images, self.labels = image_filenames, labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.labels) / float(self.batch_size)))

    def __getitem__(self, idx):
        image_batch = self.images[idx * self.batch_size:(idx + 1) * self.batch_size]
        label_batch = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        image_array = []
        for file_name in image_batch:
            img = resize(
                imread(file_name),
                (utils.image_width, utils.image_width)
            )

            image_array.append(img)
        
        return np.array(image_array), np.array(label_batch)