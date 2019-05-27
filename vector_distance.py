from __future__ import absolute_import, division, print_function

import os

import pickle
import numpy as np
from scipy import spatial

from load_dataset_cache import vectors_and_labels
from params import Params, Config
from new_utils import Consts, Dirs
from train_final_layer import read_training_session

def save_session_params(params, train_config, val_config, bias, euc_results, cos_results):
    training_session = read_training_session()

    if not os.path.exists(Dirs.LOG_DIR.format(sess=training_session)):
        os.mkdir(Dirs.LOG_DIR.format(sess=training_session))

    configs = [train_config, val_config]
    with open(Dirs.PARAMS_PATH.format(sess=training_session), "w") as file:
        file.write("Session: " + str(training_session) + "\n\n")
        file.write("Params:\n")
        file.write("\tModel: " + params.model + "\n")
        file.write("\tNumber of classes: " + str(params.num_classes) + "\n")
        file.write("\tExamples per class: " + str(params.examples_per_class) + "\n")
        file.write("\tExamples per class: " + str(params.examples_per_class) + "\n")
        file.write("\tCrop percentage: " + str(params.crop_pctg) + "\n")
        file.write("\tBias:" + str(bias) + "\n")

        for config in configs:
            file.write("\n" + config.split.capitalize() + " Config:\n")
            for version in config.list_versions:
                file.write("\tInclude" + version[1] + ": " + str(version[0]) + "\n")

        file.write("Euclidean Results:\n")
        file.write("\tAccuracy: {}\n".format(euc_results[0]))
        file.write("\tInaccuracy: {}\n".format(euc_results[1]))
        file.write("\tFalse Positive Rate: {}\n".format(euc_results[2]))
        file.write("\tFalse Negative Rate: {}\n".format(euc_results[3]))

        file.write("Cosine Results:\n")
        file.write("\tAccuracy: {}\n".format(cos_results[0]))
        file.write("\tInaccuracy: {}\n".format(cos_results[1]))
        file.write("\tFalse Positive Rate: {}\n".format(cos_results[2]))
        file.write("\tFalse Negative Rate: {}\n".format(cos_results[3]))

        file.close()

    return training_session

def calc_means(paths, labels):
    dict_by_iden = {}
    for path, label in zip(paths, labels):
        if label not in dict_by_iden.keys():
            dict_by_iden[label] = []

        with open(path, "rb") as vector:
            dict_by_iden[label].append(pickle.load(vector))

    means = {}
    for key in dict_by_iden:
        means[key] = np.nanmean(dict_by_iden[key], 0)

    return means

def vector_distance(params, train_config, val_config, bias, euclidean=True):
    train_paths, train_labels = vectors_and_labels(params, train_config)
    val_paths, val_labels = vectors_and_labels(params, val_config)

    means = calc_means(train_paths, train_labels)
    results = []

    for path, label in zip(val_paths, val_labels):
        with open(path, "rb") as vector_fd:
            vector = pickle.load(vector_fd)

            keys = []
            dists = []
            for key in means:
                keys.append(key)
                if euclidean:
                    dists.append(np.linalg.norm(means[key] - vector))
                else:
                    dists.append(spatial.distance.cosine(means[key], vector))

            normalized_distances = [float(dist)/max(dists) for dist in dists]
            distances = list(zip(keys, normalized_distances))

            distances.sort(key=lambda tup: tup[1])
            results.append((label, distances))

    correct_guesses = 0
    wrong_guesses = 0
    false_positives = 0
    false_negatives = 0
    for result in results:
        if result[1][0][1] <= bias:
            if result[0] == result[1][0][0]:
                correct_guesses += 1
            else:
                false_positives += 1
        elif result[0] == result[1][0][0]:
            false_negatives += 1
        else:
            wrong_guesses += 1

    accuracy = (correct_guesses * 100) / len(results)
    fp_rate = (false_positives * 100) / len(results)
    fn_rate = (false_negatives * 100) / len(results)
    inacuracy = (wrong_guesses * 100) / len(results)

    distance_method = "Euclidean" if euclidean else "Cosine"
    print("{} Results:".format(distance_method))
    print("\tAccuracy: {}".format(accuracy))
    print("\tInaccuracy: {}".format(inacuracy))
    print("\tFalse Positive Rate: {}".format(fp_rate))
    print("\tFalse Negative Rate: {}".format(fn_rate))

    return (accuracy, inacuracy, fp_rate, fn_rate)

def main():
    params = Params(
        model=Consts.VGG16,
        num_classes=50,
        examples_per_class=5,
        crop_pctg=5,
        include_unknown=False   # when in vector mode: always false
    )
    train_config = Config(
        split=Consts.TRAIN,
        include_original=True,
        include_transform=True,
        include_face_patch=True
    )
    val_config = Config(
        split=Consts.VAL,
        include_original=True,
        include_transform=True,
        include_face_patch=True
    )
    bias = 0.625 # bias tb e var a estudar

    euc_results = vector_distance(params, train_config, val_config, bias=bias, euclidean=True)
    cos_results = vector_distance(params, train_config, val_config, bias=bias, euclidean=False)

    save_session_params(params, train_config, val_config, bias, euc_results, cos_results)

if __name__ == "__main__":
    main()
