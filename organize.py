from __future__ import absolute_import, division, print_function

import os
import shutil

from PIL import Image
import numpy as np

from load_celeba import load_train_val_test
import utils

def load_labels_dict():
    labels_dict = {}
    with open(utils.labels_path, 'r') as f:
        for line in f:
            file_name = line.split()[0]
            label = line.split()[1]

            labels_dict[file_name] = label
    
    return labels_dict

def from_labels_to_folders():
    tvt = load_train_val_test()
    labels = load_labels_dict()

    if not os.path.exists(utils.split_dir):
        os.makedirs(utils.split_dir)

    if not os.path.exists(utils.train_dir):
        os.makedirs(utils.train_dir)

    if not os.path.exists(utils.val_dir):
        os.makedirs(utils.val_dir)

    if not os.path.exists(utils.test_dir):
        os.makedirs(utils.test_dir)
    
    if not os.path.exists(utils.train_dir_aug):
        os.makedirs(utils.train_dir_aug)

    train_examples = 0
    val_examples = 0
    test_examples = 0

    for key in tvt:
        if int(labels[key]) <= utils.num_classes:
            if tvt[key] == "0":
                if not os.path.exists(utils.train_dir + labels[key]):
                    os.makedirs(utils.train_dir + labels[key])
                shutil.copy2(utils.images_dir + key, utils.train_dir + labels[key])
                train_examples += 1
            elif tvt[key] == "1":
                if not os.path.exists(utils.val_dir + labels[key]):
                    os.makedirs(utils.val_dir + labels[key])
                shutil.copy2(utils.images_dir + key, utils.val_dir + labels[key])
                val_examples += 1
            elif tvt[key] == "2":
                if not os.path.exists(utils.test_dir + labels[key]):
                    os.makedirs(utils.test_dir + labels[key])
                shutil.copy2(utils.images_dir + key, utils.test_dir + labels[key])
                test_examples += 1
        
    print("train examples: " + str(train_examples))
    print("val examples: " + str(val_examples))
    print("test examples: " + str(test_examples))

    print("total: " + str(train_examples+val_examples+test_examples))

def from_folders_to_labels():
    with open(utils.split_dir + "split_" + str(utils.num_classes) + "_aug_labels.txt", "w") as labels:
        for folder in os.listdir(utils.split_dir):
            if os.path.isdir(utils.split_dir + folder):
                for subfolder in os.listdir(utils.split_dir + folder):
                    for file in os.listdir(utils.split_dir + folder + "/" + subfolder):
                        labels.write(file + " " + subfolder + "\n")

def get_bbox_dict():
    bbox_dict = {}
    with open(utils.raw_dir + "list_bbox_celeba.txt") as bbox:
        i = 0
        for line in bbox:
            i+=1
            if (i > 2):
                attrs = line.split()
                img_bbox = {
                    "x": int(attrs[1]),
                    "y": int(attrs[2]),
                    "width": int(attrs[3]),
                    "height": int(attrs[4]),
                }
                bbox_dict[attrs[0]] = img_bbox

    return bbox_dict

def crop_dataset():
    if not os.path.exists(utils.crop_dir):
        os.makedirs(utils.crop_dir)
    
    bbox_dict = get_bbox_dict()

    for img in os.listdir(utils.images_dir):
        bbox = (
            bbox_dict[img]["x"],
            bbox_dict[img]["y"],
            bbox_dict[img]["x"] + bbox_dict[img]["width"],
            bbox_dict[img]["y"] + bbox_dict[img]["height"]
        )
        image = Image.open(utils.images_dir + img)
        crop = image.crop(bbox)
        crop.save(utils.crop_dir + img)


    


# from_labels_to_folders()
# from_folders_to_labels()
crop_dataset()