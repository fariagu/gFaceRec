from __future__ import absolute_import, division, print_function

import os
import numpy as np

import keras

from skimage.io import imread
from skimage.transform import resize

import pickle

import utils
from load_celeba import load_vec_dict

# split_str
TRAIN = "train"
VAL = "val"

splits = [TRAIN, VAL]

cache_split = utils.cache_dir + "split_" + str(utils.num_classes) + "/"

bias = 0.0

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

def predict_raw():
    train_fv_result, val_fv_result = load_vec_dict()

    results = [train_fv_result, val_fv_result]
    results_mean = []
    results_std = []

    # [0] is train_split, [1] is val_split
    for i in range(len(splits)):
        split = results[i]
        means = {}
        stds = {}
        for iden in split:
            means[iden] = [np.nanmean(split[iden], 0)]
            stds[iden] = [np.nanstd(split[iden], 0)]
        
        results_mean.append(means)
        results_std.append(stds)

        pickle.dump(results[i], open(cache_split + splits[i] + "_fv.pkl", "wb"))
        pickle.dump(results_mean[i], open(cache_split + splits[i] + "_fv_mean.pkl", "wb"))
        pickle.dump(results_std[i], open(cache_split + splits[i] + "_fv_std.pkl", "wb"))

    return results[0], results_mean[0], results_std[0], results[1], results_mean[1], results_std[1]

def predict():
    results = []
    results_mean = []
    results_std = []

    for split_str in splits:
        if os.path.exists(cache_split + split_str + "_fv.pkl"):
            if os.path.exists(cache_split + split_str + "_fv_mean.pkl"):
                if os.path.exists(cache_split + split_str + "_fv_std.pkl"):
                    fv = pickle.load(open(cache_split + split_str + "_fv.pkl", "rb")),
                    fv_mean = pickle.load(open(cache_split + split_str + "_fv_mean.pkl", "rb")),
                    fv_std = pickle.load(open(cache_split + split_str + "_fv_std.pkl", "rb")),
                    
                    # nao sei porque e que retorna tuplo quando faco dump de um dicionario mas saf*da
                    results.append(fv[0])
                    results_mean.append(fv_mean[0])
                    results_std.append(fv_std[0])
    
    if len(results) == len(splits):
        # train_fv, train_fv_mean, train_fv_std, val_fv, val_fv_mean, val_fv_std
        return results[0], results_mean[0], results_std[0], results[1], results_mean[1], results_std[1]

    return predict_raw()

def trim_array(sorted_array, bias):
    cut_off = sorted_array[0][0] + bias
    for i, elem in enumerate(sorted_array):
        if elem[0] > cut_off:
            return sorted_array[:i]


def validate():
    # INDEXES
    DISTANCE = 0
    LABEL = 1

    train_data, train_data_mean, train_data_std, val_data, val_data_mean, val_data_std = predict()

    total_samples = 0
    unsure = 0
    accurate = 0
    inacurate = 0
    for key, value in val_data.items():
        for fv in value:
            total_samples += 1
            distances = []
            prediction = "0"
            for t_key, t_value_arr in train_data_mean.items():
            # for t_key, t_value_arr in train_data.items():
                for t_value in t_value_arr:
                    dist = np.linalg.norm(fv-t_value)
                    distances.append((dist, t_key))
            # sort by distance
            distances.sort(key=lambda tuple: tuple[0])
            prediction = distances[0][LABEL]

            distances = trim_array(distances, bias)
            
            # print("distance between " + key + " and " + t_key + ": " + str(dist))
            if len(distances) > 1 and abs(distances[0][DISTANCE] - distances[1][DISTANCE]) < bias:   #unsure
                unsure += 1
                sec_distances = []
                for sec_key, sec_value_arr in train_data.items():
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

                prediction = sec_distances[0][1]

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


validate()