from __future__ import absolute_import, division, print_function

import os

from skimage.io import imread, imsave
from skimage.transform import resize
import numpy as np
import cv2

from new_utils import Consts

# import random as rand
# import time

import keras
# from keras.preprocessing.image import ImageDataGenerator

# import utils

# def get_transform():
#     range_shift = int(15 * utils.image_width / 100)
#     range_patch = int(utils.image_width / 2)

#     theta = rand.randint(-15, 15)
#     tx = rand.randint(-range_shift, range_shift)
#     ty = rand.randint(-range_shift, range_shift)
#     flip = rand.randint(0, 3)
    
#     x_patch = rand.randint(0, range_patch)
#     y_patch = rand.randint(0, range_patch)

#     transform = {
#         "theta": theta,
#         "tx": tx,
#         "ty": ty,
#         "sheer": theta,
#         "flip_horizontal": True if flip == 0 else False
#     }

#     return transform

# def get_patched_image(img):
#     range_patch = int(utils.image_width / 3)
#     x_patch = rand.randint(0, utils.image_width - range_patch)
#     y_patch = rand.randint(0, utils.image_width - range_patch)

#     return cv2.rectangle(
#         img,
#         (x_patch, y_patch),
#         (x_patch + range_patch, y_patch + range_patch),
#         (0.9412, 0.9176, 0.8392),    #   (240, 234, 214)
#         -1
#     )

# def get_timestamp():
#     timestamp = str(time.time())
#     timestamp = timestamp.split(".")[0] + timestamp.split(".")[1] + "_"

#     return timestamp

class Generator(keras.utils.Sequence):

    def __init__(self, image_filenames, labels, batch_size, model): # , save_to_dir=False, mode=NO_AUG
        self.images, self.labels = image_filenames, labels
        self.batch_size = batch_size
        self.model = model
        # self.save_to_dir = save_to_dir
        # self.mode = mode

    def __len__(self):
        return int(np.ceil(len(self.labels) / float(self.batch_size)))

    def __getitem__(self, idx):
        image_batch = self.images[idx * self.batch_size:(idx + 1) * self.batch_size]
        label_batch = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        # im_gen = ImageDataGenerator()

        image_array = []
        for file_name in image_batch:
            img = resize(
                imread(file_name),
                (Consts.getImageSize(self.model))
            )

            # if self.mode == TRANSFORM:
            #     img = im_gen.apply_transform(img, get_transform())
            # elif self.mode = FACE_PATCH:
            #     img = get_patched_image(img)
            
            if self.model == Consts.VGG16:
                np.transpose(img, (2, 0, 1))

            image_array.append(img)
            
            # if self.save_to_dir:
            #     path = utils.created_aug_imgs
            #     if not os.path.exists(path):
            #         os.mkdir(path)
                
            #     file_name = file_name.split("/")[-1]
            #     x = path + get_timestamp() + file_name
            #     imsave(x, img)
        
        return np.array(image_array), np.array(label_batch)