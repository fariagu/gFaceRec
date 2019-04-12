from __future__ import absolute_import, division, print_function

import os
import numpy as np

import keras

from skimage.io import imread
from skimage.transform import resize

import pickle

import utils
from load_local_model import load_facenet_fv

# split_str
TRAIN = "train"
VAL = "val"

cache_split = utils.cache_dir + "split_" + str(utils.num_classes) + "/"

bias = 0.55

def load_split(split_str):
    if not os.path.exists(cache_split + split_str + "_data_mean.pkl"):
        return load_split_raw(split_str)
    else:
        return pickle.load(open(cache_split + split_str + "_data.pkl", "rb"))


def load_split_raw(split_str):
    split_dict = {}

    for folder in os.listdir(utils.split_dir + split_str):
        split_dict[folder] = []
        for file in os.listdir(utils.split_dir + split_str + "/" + folder):
            file_name = utils.split_dir + split_str + "/" + folder + "/" + file
            split_dict[folder].append(
                resize(imread(file_name),(utils.image_width, utils.image_width))
            )

    if not os.path.exists(cache_split):
        os.mkdir(cache_split)

    pickle.dump(split_dict, open(cache_split + split_str + "_data.pkl", "wb"))

    return split_dict

def predict_raw(split_str):
    # split = load_split(split_str)
    model = load_facenet_fv()

    split_fv_result = {}
    split_fv_result_mean = {}
    split_fv_result_std = {}

    key = "1"
    # for key in split:
        # data = model.predict(np.array(split[key]), verbose=1)

    # return model.predict_generator(
    #     generator=train_batch_generator,
    #     verbose=1,
    #     use_multiprocessing=utils.multiprocessing,
    #     workers=utils.n_workers,
    # )

    data = nn_train_predict(predict=True)
    split_fv_result[key] = data
    split_fv_result_mean[key] = [np.nanmean(data, 0)]
    split_fv_result_std[key] = [np.nanstd(data, 0)]
    
    pickle.dump(split_fv_result, open(cache_split + split_str + "_fv.pkl", "wb"))
    pickle.dump(split_fv_result_mean, open(cache_split + split_str + "_fv_mean.pkl", "wb"))
    pickle.dump(split_fv_result_std, open(cache_split + split_str + "_fv_std.pkl", "wb"))

    return split_fv_result, split_fv_result_mean, split_fv_result_std

def predict(split_str):
    if os.path.exists(cache_split + split_str + "_fv.pkl"):
        if os.path.exists(cache_split + split_str + "_fv_mean.pkl"):
            if os.path.exists(cache_split + split_str + "_fv_std.pkl"):
                fv = pickle.load(open(cache_split + split_str + "_fv.pkl", "rb")),
                fv_mean = pickle.load(open(cache_split + split_str + "_fv_mean.pkl", "rb")),
                fv_std = pickle.load(open(cache_split + split_str + "_fv_std.pkl", "rb")),
                
                return fv, fv_mean, fv_std
    
    return predict_raw(split_str)

def trim_array(sorted_array, bias):
    cut_off = sorted_array[0][0] + bias
    for i, elem in enumerate(sorted_array):
        if elem[0] > cut_off:
            return sorted_array[:i]


def validate():
    train_data, train_data_mean, train_data_std = predict(TRAIN)
    val_data, val_data_mean, val_data_std = predict(VAL)

    total_samples = 0
    # accurate_predictions = 0
    unsure = 0
    accurate = 0
    inacurate = 0
    for key, value in val_data[0].items():
        for fv in value:
            total_samples += 1
            distances = []
            prediction = "0"
            for t_key, t_value_arr in train_data_mean[0].items():
            # for t_key, t_value_arr in train_data.items():
                for t_value in t_value_arr:
                    dist = np.linalg.norm(fv-t_value)
                    distances.append((dist, t_key))
            # sort by distance
            distances.sort(key=lambda tuple: tuple[0])
            prediction = distances[0][1]

            distances = trim_array(distances, bias)
            
            # print("distance between " + key + " and " + t_key + ": " + str(dist))
            if len(distances) > 1 and abs(distances[0][0] - distances[1][0]) < bias:   #unsure
                unsure += 1
                sec_distances = []
                for sec_key, sec_value_arr in train_data[0].items():
                # for sec_key, sec_value_arr in train_data_mean.items():
                    for sec_value in sec_value_arr:
                        dist = np.linalg.norm(fv-sec_value)
                        sec_distances.append((dist, sec_key))
                    
                sec_distances.sort(key=lambda tuple: tuple[0])
                # sec_distances = trim_array(sec_distances, bias)

                if len(distances) > 0:
                    for dist in distances:
                        sec_dists, sec_keys = zip(*sec_distances)
                        if dist[1] in sec_keys:
                            prediction = dist[1]

                # else do nothing

                preditction = sec_distances[0][1]

            if prediction == key:
                accurate += 1
            else:
                inacurate += 1
                # print("######### Failed prediction #########")
                # print("Expected: " + key + "-- got: " + prediction)
                # print(distances)

    # accuracy = accurate_predictions * 100 / total_samples
    accuracy = accurate * 100 / total_samples
    false_positive_rate = inacurate * 100 / total_samples
    unsure_rate = unsure * 100 / total_samples
    
    print("#############################")
    print("Accuracy: " + str(accuracy))
    print("False positive rate: " + str(false_positive_rate))
    print("Unsure rate: " + str(unsure_rate))


# validate()