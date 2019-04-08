from __future__ import absolute_import, division, print_function

import os
import numpy as np

import keras

from skimage.io import imread
from skimage.transform import resize

import utils
from load_local_model import load_facenet_fv

split_dir = "C:/datasets/CelebA/split_10/"
# hard coded FOR NOW

TRAIN = "train/"
VAL = "val/"

def load_split(dir):
    train_split_dict = {}

    for folder in os.listdir(split_dir + dir):
        train_split_dict[folder] = []
        for file in os.listdir(split_dir + dir + folder):
            file_name = split_dir + dir + folder + "/" + file
            train_split_dict[folder].append(
                resize(imread(file_name),(utils.image_width, utils.image_width))
            )

    return train_split_dict

def predict(split_str, mean):
    split = load_split(split_str)
    model = load_facenet_fv()

    split_fv_result = {}

    if mean:
        for key in split:
            split_fv_result[key] = np.mean(model.predict(np.array(split[key]), verbose=1), 0)
    else:
        for key in split:
            split_fv_result[key] = model.predict(np.array(split[key]), verbose=1)

    return split_fv_result

def validate():
    train_data = predict(TRAIN, mean=True)
    val_data = predict(VAL, mean=False)

    # print(train_data["1"].shape)
    # print(val_data["1"].shape)

    total_samples = 0
    accurate_predictions = 0
    for key, value in val_data.items():
        for fv in value:
            total_samples += 1
            # print("########################")
            # distances = []
            lowest_distance = float("inf")
            prediction = "0"
            for t_key, t_value in train_data.items():
                dist = np.linalg.norm(fv-t_value)
                if dist < lowest_distance:
                    lowest_distance = dist
                    prediction = t_key
            # distances.append((dist, t_key))
            
            # print("distance between " + key + " and " + t_key + ": " + str(dist))
            if prediction == key:
                accurate_predictions += 1
            # else:
                # print("Failed prediction: " + )

    accuracy = accurate_predictions * 100 / total_samples
    print(accuracy)


validate()