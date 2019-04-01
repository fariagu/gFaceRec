from __future__ import absolute_import, division, print_function

import os
import shutil

from PIL import Image
import numpy as np

from load_celeba import load_train_val_test
from utils import split_dir, train_dir, val_dir, test_dir, images_dir, labels_path, num_classes

def load_labels_dict():
    labels_dict = {}
    with open(labels_path, 'r') as f:
        for line in f:
            file_name = line.split()[0]
            label = line.split()[1]

            labels_dict[file_name] = label
    
    return labels_dict

def from_labels_to_folders():
    tvt = load_train_val_test()
    labels = load_labels_dict()

    if not os.path.exists(split_dir):
        os.makedirs(split_dir)

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    train_examples = 0
    val_examples = 0
    test_examples = 0

    for key in tvt:
        if int(labels[key]) <= num_classes:
            if tvt[key] == "0":
                if not os.path.exists(train_dir + labels[key]):
                    os.makedirs(train_dir + labels[key])
                shutil.copy2(images_dir + key, train_dir + labels[key])
                train_examples += 1
            elif tvt[key] == "1":
                if not os.path.exists(val_dir + labels[key]):
                    os.makedirs(val_dir + labels[key])
                shutil.copy2(images_dir + key, val_dir + labels[key])
                val_examples += 1
            elif tvt[key] == "2":
                if not os.path.exists(test_dir + labels[key]):
                    os.makedirs(test_dir + labels[key])
                shutil.copy2(images_dir + key, test_dir + labels[key])
                test_examples += 1
        
    print("train examples: " + str(train_examples))
    print("val examples: " + str(val_examples))
    print("test examples: " + str(test_examples))

    print("total: " + str(train_examples+val_examples+test_examples))

def from_folders_to_labels():
    with open(split_dir + "split_" + str(num_classes) + "_aug_labels.txt", "w") as labels:
        for folder in os.listdir(split_dir):
            if os.path.isdir(split_dir + folder):
                for subfolder in os.listdir(split_dir + folder):
                    for file in os.listdir(split_dir + folder + "/" + subfolder):
                        labels.write(file + " " + subfolder + "\n")

# from_labels_to_folders()
from_folders_to_labels()