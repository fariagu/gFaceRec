from __future__ import absolute_import, division, print_function

import numpy as np
from scipy import spatial
import pickle

from load_dataset_cache import vectors_and_labels
from params import Params, Config
from new_utils import Consts

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

def vector_distance(params, train_config, val_config, euclidean=True):
    train_paths, train_labels = vectors_and_labels(params, train_config)
    val_paths, val_labels = vectors_and_labels(params, val_config)

    means = calc_means(train_paths, train_labels)
    results = []

    for path, label in zip(val_paths, val_labels):
        with open(path, "rb") as vector_fd:
            vector = pickle.load(vector_fd)

            distances = []
            for key in means:
                if euclidean:
                    distances.append((key, np.linalg.norm(means[key] - vector)))
                else:
                    spatial.distance.cosine(means[key], vector)

            distances.sort(key=lambda tup: tup[1])
            results.append((label, distances))

    correct_guesses = 0
    for result in results:
        if result[0] == result[1][0][0]:
            correct_guesses += 1

    accuracy = (correct_guesses * 100) / len(results)

    print("Accuracy: {}".format(accuracy))


def main():
    params = Params(
        model=Consts.INCEPTIONV3,
        num_classes=10,
        examples_per_class=1,
        crop_pctg=20,
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

    vector_distance(params, train_config, val_config)

if __name__ == "__main__":
    main()
