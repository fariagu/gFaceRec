from __future__ import absolute_import, division, print_function

import os
import numpy as np

import keras

from skimage.io import imread
from skimage.transform import resize

import utils
from load_local_model import load_facenet_fv

TRAIN = "train/"
VAL = "val/"

bias = 1.0

def load_split(dir):
    train_split_dict = {}

    for folder in os.listdir(utils.split_dir + dir):
        train_split_dict[folder] = []
        for file in os.listdir(utils.split_dir + dir + folder):
            file_name = utils.split_dir + dir + folder + "/" + file
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

    total_samples = 0
    accurate_predictions = 0
    unsure = 0
    accurate_and_sure = 0
    inacurate_but_sure = 0
    for key, value in val_data.items():
        for fv in value:
            total_samples += 1
            distances = []
            lowest_distance = float("inf")
            prediction = "0"
            for t_key, t_value in train_data.items():
                dist = np.linalg.norm(fv-t_value)
                if dist < lowest_distance:
                    lowest_distance = dist
                    # prediction = t_key
                distances.append((dist, t_key))
            # sort by distance
            distances.sort(key=lambda tuple: tuple[0])
            prediction = distances[0][1]
            
            # print("distance between " + key + " and " + t_key + ": " + str(dist))
            if abs(distances[0][0] - distances[1][0]) < bias:   #unsure
                unsure += 1
            else:
                if prediction == key:
                    # accurate_predictions += 1
                    accurate_and_sure += 1
                else:
                    inacurate_but_sure += 1
                    # print("######### Failed prediction #########")
                    # print("Expected: " + key + "-- got: " + prediction)
                    # print(distances)

    # accuracy = accurate_predictions * 100 / total_samples
    accuracy = accurate_and_sure * 100 / total_samples
    false_positive_rate = inacurate_but_sure * 100 / total_samples
    unsure_rate = unsure * 100 / total_samples
    
    print("#############################")
    print("Accuracy: " + str(accuracy))
    print("False positive rate: " + str(false_positive_rate))
    print("Unsure rate: " + str(unsure_rate))


validate()