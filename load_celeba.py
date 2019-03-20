from __future__ import absolute_import, division, print_function

import os
import concurrent.futures

import random as rand

import gc

from PIL import Image
import numpy as np

size = 128, 128

# windows
raw_dir = "C:/datasets/CelebA/img_align_celeba/"
cache_dir = "C:/dataset_cache/CelebA/"

# rhel7 (grid.fe.up.pt)
raw_dir = "/homes/up201304501/CelebA/img_align_celeba/"
cache_dir = "C:/dataset_cache/CelebA/"

"""
    não faz muito porque neste dataset so tenho labels numericas
    (o conteudo do ficheiro vai ser literalmente: 0\n1\n2\n3\n4\n...n)
    NOTA: labels neste ficheiro estao desfazadas uma unidade (as labels para o keras comecam no 0,
            nas anotacoes do dataset celebA comecam no 1)
"""
def write_string_labels(labels):
    path = cache_dir +  "labels.txt"

    with open(path, 'w') as f:
        for i in range(max(labels)):
            if i == range(max(labels) - 2):
                f.write(str(i))
            else:
                f.write(str(i) + "\n")

def get_num_classes():
    path = cache_dir + "labels.txt"

    with open(path, 'r') as f:
        i = 0
        for line in f:
            i+=1
        
        return i
    
    return -1

def load_train_val_test(random=True):
    train_val_test = {}
    train_samples = 0
    val_samples = 0
    test_samples = 0
    path = "C:/datasets/CelebA/list_eval_partition.txt"
    with open(path, 'r') as f:
        for line in f:
            tmp = line.split()

            if random:
                seed = rand.randint(1, 20)
                if seed == 20:
                    train_val_test[tmp[0]] = "2"
                    test_samples+=1
                elif seed == 19:
                    train_val_test[tmp[0]] = "1"
                    val_samples+=1
                else:
                    train_val_test[tmp[0]] = "0"
                    train_samples+=1
            else:
                train_val_test[tmp[0]] = tmp[1]
    
    return train_val_test, train_samples, val_samples, test_samples

def load_labels():
    labels = {}
    path = "C:/datasets/CelebA/identity_CelebA.txt"
    with open(path, 'r') as f:
        for line in f:
            tmp = line.split()
            labels[tmp[0]] = int(tmp[1])
    
    return labels

def load_image_filenames_and_labels():
    train_images = []
    train_labels = []
    val_images = []
    val_labels = []
    test_images = []
    test_labels = []

    train_v_test, train_len, val_len, test_len = load_train_val_test()

    path = "C:/datasets/CelebA/identity_CelebA.txt"
    with open(path, 'r') as f:
        for line in f:
            file_name = line.split()[0]
            label = line.split()[1]

            # way of specifying which subsset of the dataset being used for training
            if int(label) < 100:

                if train_v_test[file_name] == "0":
                    train_images.append(raw_dir + file_name)
                    train_labels.append(int(label)-1)
                elif train_v_test[file_name] == "1":
                    val_images.append(raw_dir + file_name)
                    val_labels.append(int(label)-1)
                else:
                    test_images.append(raw_dir + file_name)
                    test_labels.append(int(label)-1)
    
    return train_images, train_labels, val_images, val_labels, test_images, test_labels
