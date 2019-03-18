from __future__ import absolute_import, division, print_function

import os
import concurrent.futures

import random as rand

from PIL import Image
import numpy as np

size = 128, 128

raw_dir = "C:/datasets/CelebA/img_align_celeba/"
cache_dir = "C:/dataset_cache/CelebA/"

def load_image(file_path):
    img = Image.open(file_path)
    img.resize(size)
    img.load()
    data = np.asarray(img, dtype="uint8")
    return data

"""
não faz muito porque neste dataset so tenho labels numericas
(o conteudo do ficheiro vai ser literalmente: 0\n1\n2\n3\n4\n...n)
"""
def write_string_labels(labels):
    path = cache_dir +  "labels.txt"

    with open(path, 'w') as f:
        for i in range(max(labels)):
            f.write(str(i+1) + "\n")

def load_test_v_train(random=True):
    train_v_test = {}
    path = "C:/datasets/CelebA/list_eval_partition.txt"
    with open(path, 'r') as f:
        for line in f:
            tmp = line.split()

            if random:
                if (rand.randint(1, 20)) == 20:
                    train_v_test[tmp[0]] = "1"
                else:
                    train_v_test[tmp[0]] = "0"
            else:
                train_v_test[tmp[0]] = tmp[1]
    
    return train_v_test

def load_labels():
    labels = {}
    path = "C:/datasets/CelebA/identity_CelebA.txt"
    with open(path, 'r') as f:
        for line in f:
            tmp = line.split()
            labels[tmp[0]] = int(tmp[1])
    
    return labels

def load_raw_dataset():
    train_images = []
    test_images = []
    train_labels = []
    test_labels = []
    str_labels = []

    # ler ficheiro para ver se é train ou test "list_eval_partition.txt"
    test_v_train = load_test_v_train()
    labels = load_labels()

    for root, dirs, files in os.walk(raw_dir):
        for file in files:

            if test_v_train[file] == '0':
                train_images.append(load_image(root + file))
                train_labels.append(labels[file])
            else:
                test_images.append(load_image(root + file))
                test_labels.append(labels[file])
        
    np_train_images = np.array(train_images)
    np_train_labels = np.array(train_labels)
    np_test_images = np.array(test_images)
    np_test_labels = np.array(test_labels)

    train_images_cache = cache_dir + "train_images"
    train_labels_cache = cache_dir + "train_labels"
    test_images_cache = cache_dir + "test_images"
    test_labels_cache = cache_dir + "test_labels"
    np.savez_compressed(train_images_cache, np_train_images)
    np.savez_compressed(train_labels_cache, np_train_labels)
    np.savez_compressed(test_images_cache, np_test_images)
    np.savez_compressed(test_labels_cache, np_test_labels)
    write_string_labels(labels.values())


    return (np_train_images, np_train_labels), (np_test_images, np_test_labels)

def load_npy_dataset():
    train_images_path = cache_dir + "train_images.npz"
    train_labels_path = cache_dir + "train_labels.npz"
    test_images_path = cache_dir + "test_images.npz"
    test_labels_path = cache_dir + "test_labels.npz"
    train_images_npz = np.load(train_images_path)
    train_labels_npz = np.load(train_labels_path)
    test_images_npz = np.load(test_images_path)
    test_labels_npz = np.load(test_labels_path)

    train_images = train_images_npz[train_images_npz.files[0]]
    train_labels = train_labels_npz[train_labels_npz.files[0]]
    test_images = test_images_npz[test_images_npz.files[0]]
    test_labels = test_labels_npz[test_labels_npz.files[0]]
    
    return (train_images, train_labels), (test_images, test_labels)

def load_dataset():
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        return load_raw_dataset()
    else:
        return load_npy_dataset()
    
def main():
    (train_images, train_labels), (test_images, test_labels) = load_dataset()

    print(train_images.shape)
    print(test_images.shape)
    # load_raw_dataset()
    # load_npy_dataset()

main()