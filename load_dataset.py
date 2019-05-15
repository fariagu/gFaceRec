from __future__ import absolute_import, division, print_function

import os
import platform

from shutil import copyfile
from multiprocessing import Pool
from functools import partial
from random import shuffle

import numpy as np
import pickle

from generate_augmented_data import augment_image
from detect_and_crop import detect_and_crop

CELEBA_NUM_IDENTITIES = 10177

def identity_dict(filepath):
    """ Parses identity text file into dictionary
    
    Arguments:
        filepath {string} -- identity_CelebA.txt path (ex.: "/home/user/identity_CelebA.txt")
    
    Returns:
        dictionary -- keys are the pictures' filenames and values are their respective identity (starting at 0)
    """

    """ INDEXES """
    FILENAME    = 0
    LABEL       = 1

    if not os.path.exists(filepath):
        print("USER ERROR: invalid filepath value.")
        return

    identity_dict = {}

    with open(filepath, 'r') as labels:
        for line in labels:
            line_split = line.split()

            identity_dict[line_split[FILENAME]] = int(line_split[LABEL])-1

    return identity_dict

def filter_identity_dict(identity_dict, num_classes):
    """ Removes unnecessary keys from dictionary
    
    Arguments:
        identity_dict {dictionary} -- {"filename": int(label)}
        num_classes {integer} -- number of classes allowed to remain in the dictionary
    
    Returns:
        dictionary -- version of identity_dict after removing unwanted entries
    """

    filtered_identity_dict = {}
    for key in identity_dict.keys():
        if identity_dict[key] < num_classes:
            filtered_identity_dict[key] = identity_dict[key]
    
    return filtered_identity_dict

def labels_by_class_dict(labels_dict):
    """ In a sense, switches keys and values from labels_dict dictionary but since
    that way the new keys wouldn't be unique, values from repeated keys are appended to a list
    
    Arguments:
        label_dict {dictionary} -- {"filename": int(label)}
    
    Returns:
        dictionary -- {str(int(label)): [image_0, ..., image_n]}
    """

    labels_by_class_dict = {}
    for key in labels_dict.keys():
        if labels_dict[key] not in labels_by_class_dict.keys():
            labels_by_class_dict[labels_dict[key]] = []
        
        labels_by_class_dict[labels_dict[key]].append(key)

    return labels_by_class_dict

def filter_celeba_identities(images_path, labels_dict, output_dir, num_identities):
    """ CelebA has 200k images os +10k identities, not all are necessary and require a lot of resources,
    so this function serves as a way of cutting out the fat
    
    Arguments:
        images_path {string} -- extracted CelebA dataset images location (slash terminated)
        labels_dict {dictionary} -- {"filename": int(value)} (NOT FILTERED)
        output_dir {string} -- where generated images will be located (slash terminated)
        num_identities {integer} -- number of identities to be used in future calculations
    """

    if not os.path.exists(images_path):
        print("USER ERROR: invalid images_path value.")
        return

    if os.path.exists(output_dir):
        if len(os.listdir(output_dir)) > 0:
            print("USER ERROR: output_dir already exists and is not empty.")
            return
    else:
        os.mkdir(output_dir)

    if num_identities < 1 or num_identities > CELEBA_NUM_IDENTITIES:
        print("USER ERROR: num_identities must be between 1 and {identities}".format(identities=CELEBA_NUM_IDENTITIES))
        return

    for file in os.listdir(images_path):
        if labels_dict[file] < num_identities:
            copyfile(images_path + file, output_dir + file)

def organize_folder(images_path, labels_dict):
    """ Reorganize pictures into subdirectories where each subdirectory corresponds to a class label
    (https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8)
    
    Arguments:
        images_path {string} -- extracted CelebA dataset images location (full or only N identities) (slash terminated)
        labels_dict {dictionary} -- {"filename": int(value)} (CAN BE PRE FILTERED)
    """
    
    if not os.path.exists(images_path):
        print("USER ERROR: invalid images_path value.")
        return
    
    for file in os.listdir(images_path):
        sub_dir =  "{}{}/".format(images_path, str(labels_dict[file]))
        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)
        
        os.rename(images_path + file, sub_dir + file)

def batch_detect_and_crop(crop_dirs):
    """ Crops pictures into different margins around the detected face (yes, also detects faces)
    
    Arguments:
        crop_dirs {list} -- location for the no_crop folder (slash terminated), and all the different crop percentages (NO_CROP must be index 0)
    """

    """ INDEXES """
    NO_CROP     = 0
    WITH_CROP   = 1

    src_dir = crop_dirs[NO_CROP]
    images = os.listdir(src_dir)

    for crop_dir in crop_dirs:
        if crop_dir != src_dir:
            percentage = int(crop_dir.split("_")[-1][:-1])

            if not os.path.exists(crop_dir):
                os.mkdir(crop_dir)

            p = Pool(8)
            func = partial(detect_and_crop, src_dir=src_dir, dst_dir=crop_dir, percentage=percentage)
            p.map(func, images)
            p.close()
            p.join()

def train_val_split(base_dir, val_percentage, labels_dict):
    """ Splits files in directory into their respective sub folder (train/ or val/)
    by taking into account val_percentage and number of photos per identity,
    making sure there is at least one validation example per class.
    
    Arguments:
        base_dir {string} -- folder where pictures reside
        val_percentage {integer} -- value between 0 and 100
        labels_dict {dictionary} -- {"filename": int(value)} (MUST BE PRE FILTERED)
    """

    if val_percentage < 0 or val_percentage > 100:
        print("USER ERROR: val_percentage must be between 0 and 100")
        return

    labels_dict_by_class = labels_by_class_dict(labels_dict)

    train_split = []
    val_split = []
    for key in labels_dict_by_class.keys():
        if len(labels_dict_by_class[key]) == 0:
            pass
        elif len(labels_dict_by_class[key]) <= 10:
            val_split.append(labels_dict_by_class[key][0])
            train_split.extend(labels_dict_by_class[key][1:])
        else:
            limit = int(len(labels_dict_by_class[key]) * val_percentage / 100)

            val_split.extend(labels_dict_by_class[key][:limit])
            train_split.extend(labels_dict_by_class[key][limit:])
    

    train_dir = base_dir + "train/"
    val_dir = base_dir + "val/"

    sub_dirs = [train_dir, val_dir]

    for sub_dir in sub_dirs:
        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)
            os.mkdir(sub_dir + "original/")
            os.mkdir(sub_dir + "transform/")
            os.mkdir(sub_dir + "face_patch/")

    for file in os.listdir(base_dir):
        if os.path.isfile(base_dir + file):
            if file in val_split:
                os.rename(base_dir + file, val_dir + "original/" + file)
            else:
                os.rename(base_dir + file, train_dir + "original/" + file)

def load_image_filenames_and_labels(filtered_dir, crop_percentage, original_train=False, original_val=False, transform_train=False, transform_val=False, face_patch_train=False, face_patch_val=False):
    """For the given crop_percentage, generates lists of filenames and labels of images
    to be posteriorly fed into a generator

    NOTE: all keyword arguments default to False so only explicitly True arguments show up on function call
        ex.: load_image_filenames_and_labels(_, _, original_train=True, original_val=True)
    
    Arguments:
        filtered_dir {string} -- path to the directory where the several cropped copies reside
        crop_percentage {integer} -- must be one of the preexisting cropped percentages or negative for "no_crop"
    
    Keyword Arguments:
        original_train {bool} -- whether to include original examples in training split (default: {False})
        original_val {bool} -- whether to include original examples in validation split (default: {False})
        transform_train {bool} -- whether to include transformed examples in training split (default: {False})
        transform_val {bool} -- whether to include transformed examples in validation split (default: {False})
        face_patch_train {bool} -- whether to include occluded examples in training split (default: {False})
        face_patch_val {bool} -- whether to include occluded examples in validation split (default: {False})
    
    Returns:
        list(string), list(string), list(string), list(string) -- 
    """
    
    train_tuples = []
    val_tuples = []
    tuples = [train_tuples, val_tuples]

    base_dir = filtered_dir

    if crop_percentage < 0:
        base_dir += "no_crop/"
    else:
        base_dir += "crop_{pctg:02d}/".format(pctg=crop_percentage)
        
        if not os.path.exists(base_dir):
            print("USER ERROR: crop_percentage value is not valid.")
    
    if original_train == False and transform_train == False and face_patch_train == False:
        print("USER ERROR: must have at least one option for training.")
        return
    
    if original_val == False and transform_val == False and face_patch_val == False:
        print("USER ERROR: must have at least one option for validation.")
        return
    
    train_og = "{}train/original/".format(base_dir)
    train_transform = "{}train/transform/".format(base_dir)
    train_face_patch = "{}train/face_patch/".format(base_dir)
    val_og = "{}val/original/".format(base_dir)
    val_transform = "{}val/transform/".format(base_dir)
    val_face_patch = "{}val/face_patch/".format(base_dir)

    train_dirs = []
    val_dirs = []

    if original_train:
        train_dirs.append(train_og)
    if transform_train:
        train_dirs.append(train_transform)
    if face_patch_train:
        train_dirs.append(train_face_patch)
    
    if original_val:
        val_dirs.append(val_og)
    if transform_val:
        val_dirs.append(val_transform)
    if face_patch_val:
        val_dirs.append(val_face_patch)

    dirs = [train_dirs, val_dirs]

    for i in range(len(tuples)):
        x = dirs[i]
        for dir in dirs[i]:
            for identity in os.listdir(dir):
                iden_dir = "{}{}/".format(dir, identity)
                for file in os.listdir(iden_dir):
                    tuples[i].append(("{}{}".format(iden_dir, file), identity))

    shuffle(train_tuples)
    shuffle(val_tuples)

    train_filenames, train_labels = zip(*train_tuples)
    val_filenames, val_labels = zip(*val_tuples)

    return train_filenames, train_labels, val_filenames, val_labels

def generate_augmentation(filtered_dir, aug_mult):
    TRANSFORM   = 0
    FACE_PATCH  = 1
    
    for dir in os.listdir(filtered_dir):
        source_dirs = [
            "{}{}/train/original/".format(filtered_dir, dir),
            "{}{}/val/original/".format(filtered_dir, dir)
        ]

        for sub_dir in source_dirs:

            for iden_dir in os.listdir(sub_dir):
                iden_dir_path = "{}{}/".format(sub_dir, iden_dir)
                
                for file_name in os.listdir(iden_dir_path):
                    file_name = "{}{}".format(iden_dir_path, file_name)

                    for i in range(aug_mult):
                        augment_image(file_name, TRANSFORM, version=i)
                        augment_image(file_name, FACE_PATCH, version=i)
            
            pass


##############################################################
##############################################################

def main():
    num_classes = 3 # temp on windows (should be 500)
    aug_mult = 10
    val_percentage = 10
    root_dir = "C:/" if platform.system() == "Windows" else "/home/gustavoduartefaria/"
    dataset_dir = "{}datasets/CelebA/".format(root_dir)
    base_dir = "{}img_celeba/".format(dataset_dir)
    filtered_dir = "{}filter_{}_classes/".format(dataset_dir, num_classes)

    if not os.path.exists(filtered_dir):
        os.mkdir(filtered_dir)
    
    crop_pctgs = [0, 5, 10, 15, 20, 25, 30]
    cropped_dirs = [
        "{}no_crop/".format(filtered_dir),
    ]
    for pctg in crop_pctgs:
        cropped_dirs.append("{}crop_{pctg:02d}/".format(filtered_dir, pctg=pctg))

    
    structured_dir = "{}test_structured_dir/".format(dataset_dir)

    labels_dict = identity_dict("{}identity_CelebA.txt".format(dataset_dir))

    # faz isto para ter so 1000 classes no maximo
    filter_celeba_identities(base_dir, labels_dict, cropped_dirs[0], num_classes)

    labels_dict = filter_identity_dict(labels_dict, num_classes)

    # depois falta fazer crops as fotos (margin: 0, 5, 10, 15, 20, 25, 30)
    batch_detect_and_crop(cropped_dirs)
    
    for dir in cropped_dirs:
        # e depois falta dividir em train/val (90/10)
        train_val_split(dir, 10, labels_dict)

        # e so depois dividir em subpastas (uma pasta por iden)
        sub_dirs = [
            "{}train/original/".format(dir),
            "{}val/original/".format(dir)
        ]

        for sub_dir in sub_dirs:
            organize_folder(sub_dir, labels_dict)
    
    generate_augmentation(filtered_dir, aug_mult)

if __name__ == "__main__":
    # load_image_filenames_and_labels(
    #     "C:/datasets/CelebA/filter_10_classes/",
    #     0,
    #     original_train=True,
    #     original_val=True,
    # )

    # generate_augmentation("C:/datasets/CelebA/filter_10_classes/", 10)

    main()