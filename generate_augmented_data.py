from __future__ import absolute_import, division, print_function

import os

import random as rand

from skimage.io import imread, imsave
import cv2

from keras.preprocessing.image import ImageDataGenerator

TRANSFORM = 0
FACE_PATCH = 1

def get_transform(width):
    range_shift_x = int(15 * width / 100)
    range_shift_y = int(15 * width / 100)
    range_patch = int(width / 2)

    theta = rand.randint(-15, 15)
    tx = rand.randint(-range_shift_x, range_shift_x)
    ty = rand.randint(-range_shift_y, range_shift_y)
    flip = rand.randint(0, 3)

    transform = {
        "theta": theta,
        "tx": tx,
        "ty": ty,
        # "sheer": theta,
        "flip_horizontal": bool(flip == 0)
    }

    return transform

def get_patched_image(img, height, width):
    range_patch_x = int(width / 3)
    range_patch_y = int(height / 3)
    x_patch = rand.randint(0, width - range_patch_x)
    y_patch = rand.randint(0, width - range_patch_y)

    return cv2.rectangle(
        img,
        (x_patch, y_patch),
        (x_patch + range_patch_x, y_patch + range_patch_y),
        # (0.9412, 0.9176, 0.8392),
        (240, 234, 214),
        -1
    )

def augment_image(file_path, mode, version):
    img = imread(file_path)

    height = img.shape[0]
    width = img.shape[1]

    im_gen = ImageDataGenerator()

    if mode == TRANSFORM:
        img = im_gen.apply_transform(img, get_transform(width))
    elif mode == FACE_PATCH:
        img = get_patched_image(img, height, width)

    mode_str = "transform/" if mode == TRANSFORM else "face_patch/"

    path_split = file_path.split("/")
    file_name = path_split[-1].split(".")[0]

    dest_dir = ""
    for dir_name in path_split[:-3]:
        dest_dir += dir_name + "/"
    dest_dir += mode_str + path_split[-2] + "/"

    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    dest_path = "{}{}_{v:02d}.jpg".format(dest_dir, file_name, v=version)
    imsave(dest_path, img)
