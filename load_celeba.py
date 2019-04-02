from __future__ import absolute_import, division, print_function

import os
# import concurrent.futures

import random as rand
# import gc
# from PIL import Image
import numpy as np
import pickle

import utils

"""
    não faz muito porque neste dataset so tenho labels numericas
    (o conteudo do ficheiro vai ser literalmente: 0\n1\n2\n3\n4\n...n)
    NOTA: labels neste ficheiro estao desfazadas uma unidade (as labels para o keras comecam no 0,
            nas anotacoes do dataset celebA comecam no 1)
    So necessario quando for feita a conversao para tflite
"""
def write_string_labels(n):
    path = utils.cache_dir +  "labels.txt"

    with open(path, 'w') as f:
        for i in range(n):
            if i == n - 1:
                f.write(str(i))
            else:
                f.write(str(i) + "\n")

# ignore for now
def get_num_classes():
    path = utils.cache_dir + "labels.txt"

    with open(path, 'r') as f:
        i = 0
        for line in f:
            i+=1
        
        return i
    
    return -1

def load_train_val_test_from_txt(random=True):

    train_val_test = {}
    with open(utils.partition_path, 'r') as f:
        for line in f:
            tmp = line.split()

            if random:
                # 80% train, 10% validation, 10% test
                seed = rand.randint(1, 20)
                if seed == 20:
                    train_val_test[tmp[0]] = "2"
                elif seed == 19:
                    train_val_test[tmp[0]] = "1"
                else:
                    train_val_test[tmp[0]] = "0"
            else:
                train_val_test[tmp[0]] = tmp[1]
    
    pickle.dump(train_val_test, open(utils.cache_partition_path, "wb"))
    
    return train_val_test

def load_train_val_test():

    if os.path.exists(utils.cache_partition_path):
        print("Loading splits from cache ...")
        return pickle.load(open(utils.cache_partition_path, "rb"))
    else:
        print("Loading splits from raw files ...")
        return load_train_val_test_from_txt()

def load_image_filenames_and_labels_from_txt():
    print("Loading images from raw data ...")

    train_images = []
    train_labels = []
    val_images = []
    val_labels = []

    train_val_test = load_train_val_test()

    with open(utils.labels_path, 'r') as f:
        for line in f:
            file_name = line.split()[0]
            label = line.split()[1]

            # way of specifying what subset of the dataset being used for training
            if int(label) < utils.num_classes:
            # if True:

                # 0: train data
                # 1: validation data
                # 2: test data
                if train_val_test[file_name] == "0":
                    train_images.append(utils.images_dir + file_name)
                    train_labels.append(int(label)-1)

                elif train_val_test[file_name] == "1":
                    val_images.append(utils.images_dir + file_name)
                    val_labels.append(int(label)-1)

    pickle.dump(train_images, open(utils.cache_dir + "train_images.pkl", "wb"))
    pickle.dump(train_labels, open(utils.cache_dir + "train_labels.pkl", "wb"))
    pickle.dump(val_images, open(utils.cache_dir + "val_images.pkl", "wb"))
    pickle.dump(val_labels, open(utils.cache_dir + "val_labels.pkl", "wb"))
    
    return train_images, train_labels, val_images, val_labels

def load_image_filenames_and_labels_from_pkl():
    print("Loading images from cache ...")

    train_images = pickle.load(open(utils.cache_dir + "train_images.pkl", "rb"))
    train_labels = pickle.dump(open(utils.cache_dir + "train_labels.pkl", "rb"))
    val_images = pickle.dump(open(utils.cache_dir + "val_images.pkl", "rb"))
    val_labels = pickle.dump(open(utils.cache_dir + "val_labels.pkl", "rb"))
    
    return train_images, train_labels, val_images, val_labels

def load_image_filenames_and_labels():
    if os.path.exists(utils.cache_dir + "train_images"):
        if os.path.exists(utils.cache_dir + "train_labels"):
            if os.path.exists(utils.cache_dir + "val_images"):
                if os.path.exists(utils.cache_dir + "val_labels"):
                    return load_image_filenames_and_labels_from_pkl()

    return load_image_filenames_and_labels_from_txt()

def load_test_data_from_txt():
    print("Loading images from raw data ...")

    test_images = []
    test_labels = []

    train_val_test = load_train_val_test()

    with open(utils.labels_path, 'r') as f:
        for line in f:
            file_name = line.split()[0]
            label = line.split()[1]

            # way of specifying what subset of the dataset being used for training
            if int(label) < utils.num_classes:
            # if True:

                # 0: train data
                # 1: validation data
                # 2: test data
                if train_val_test[file_name] == "2":
                    test_images.append(utils.images_dir + file_name)
                    test_labels.append(int(label)-1)

    pickle.dump(test_images, open(utils.cache_dir + "test_images.pkl", "wb"))
    pickle.dump(test_labels, open(utils.cache_dir + "test_labels.pkl", "wb"))
    
    return test_images, test_labels

def load_test_data_from_pkl():
    print("Loading images from cache ...")

    test_images = pickle.load(open(utils.cache_dir + "test_images.pkl", "rb"))
    test_labels = pickle.dump(open(utils.cache_dir + "test_labels.pkl", "rb"))
    
    return test_images, test_labels

def load_test_data():
    if os.path.exists(utils.cache_dir + "test_images"):
        if os.path.exists(utils.cache_dir + "test_labels"):
            return load_test_data_from_pkl()
    
    return load_test_data_from_txt()