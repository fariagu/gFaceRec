from __future__ import absolute_import, division, print_function

from skimage.io import imread
from skimage.transform import resize
import numpy as np
import cv2

import random as rand

import keras
from keras.preprocessing.image import ImageDataGenerator

import utils

def get_transform():
    range_shift = int(15 * utils.image_width / 100)
    range_patch = int(utils.image_width / 2)

    theta = rand.randint(-15, 15)
    tx = rand.randint(-range_shift, range_shift)
    ty = rand.randint(-range_shift, range_shift)
    flip = rand.randint(0, 3)
    
    x_patch = rand.randint(0, range_patch)
    y_patch = rand.randint(0, range_patch)


    # meter aqui noise 

    transform = {
        "theta": theta,
        "tx": tx,
        "ty": ty,
        "flip_horizontal": True if flip == 0 else False
    }

    return transform

def get_patched_image(img):
    range_patch = int(utils.image_width / 3)
    x_patch = rand.randint(0, utils.image_width - range_patch)
    y_patch = rand.randint(0, utils.image_width - range_patch)

    return cv2.rectangle(img, (x_patch, y_patch), (x_patch + range_patch, y_patch + range_patch), (240, 234, 214), -1)

class Generator(keras.utils.Sequence):

    def __init__(self, image_filenames, labels, batch_size):
        self.images, self.labels = image_filenames, labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.labels) / float(self.batch_size)))

    def __getitem__(self, idx):
        image_batch = self.images[idx * self.batch_size:(idx + 1) * self.batch_size]
        label_batch = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        im_gen = ImageDataGenerator()

        image_array = []
        for file_name in image_batch:
            img = resize(
                imread(file_name),
                (utils.image_width, utils.image_width)
            )

            if utils.AUGMENTATION:
                # 25% chance of shift, rotate or flip
                # 25% chance of patch
                # 50% chance of no augmentation
                seed = rand.randint(0, 3)

                if seed == 0:
                    img = im_gen.apply_transform(img, get_transform())
                elif seed == 1:
                    img = get_patched_image(img)
                # else: do nothing
            
            image_array.append(img)
        
        return np.array(image_array), np.array(label_batch)