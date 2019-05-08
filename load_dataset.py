from __future__ import absolute_import, division, print_function

import os
from shutil import copyfile
from multiprocessing import Pool
from functools import partial

import numpy as np
import pickle

import utils
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

def crop_celeba_identities(images_path, labels_dict, output_dir, num_identities):
    """ CelebA has 200k images os +10k identities, not all are necessary and require a lot of resources,
    so this function serves as a way of cutting out the fat
    
    Arguments:
        images_path {string} -- extracted CelebA dataset images location (slash terminated)
        labels_dict {dictionary} -- {"filename": int(value)}
        output_dir {string} -- location of generated images (slash terminated)
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

def import_celeba_from_base_folder(images_path, output_dir, labels_dict):
    """ Base function for structuring CelebA data after download from web, or after being <i>cropped</i>
    (crop of identities, not of pictures)
    (https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8)
    
    Arguments:
        images_path {string} -- extracted CelebA dataset images location (full or only N identities) (slash terminated)
        output_dir {string} -- location of generated images (slash terminated)
        labels_dict {dictionary} -- {"filename": int(value)}
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
    
    for file in os.listdir(images_path):
        sub_dir = output_dir + str(labels_dict[file]) + "/"
        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)
        
        copyfile(images_path + file, sub_dir + file)

def batch_detect_and_crop(src_dir):

    folders = src_dir.split("/")
    foldersmenosum = folders[:-2]
    parent_dir = ""
    for folder in foldersmenosum:
        parent_dir += folder + "/"

    images = os.listdir(src_dir)

    crop_pctgs = [0, 5, 10, 15, 20, 25, 30]

    for percentage in crop_pctgs:
        
        dst_dir = parent_dir + "crop_{pctg:02d}/".format(pctg=percentage)

        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)

        p = Pool(2)
        func = partial(detect_and_crop, src_dir, dst_dir, percentage)
        p.map(func, images)
        p.close()
        p.join()

##############################################################
##############################################################

num_classes = 1000
base_dir = "C:/datasets/CelebA/test/"
cropped_dir = "C:/datasets/CelebA/test_output_dir/no_crop/"

structured_dir = "C:/datasets/CelebA/test_structured_dir/"

labels_dict = identity_dict("C:/datasets/CelebA/identity_CelebA.txt")

# # faz isto para ter so 1000 classes no maximo
# crop_celeba_identities(base_dir, labels_dict, cropped_dir, num_classes)

# # depois falta fazer crops as fotos (margin: 0, 5, 10, 15, 20, 25, 30)
batch_detect_and_crop(cropped_dir)

# e depois falta dividir em train/val (90/10)

# e depois gerar augs

# # e so depois dividir em subpastas (uma pasta por iden)
# import_celeba_from_base_folder(cropped_dir, structured_dir, labels_dict)