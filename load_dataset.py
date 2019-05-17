from __future__ import absolute_import, division, print_function

import os

from shutil import copyfile
from multiprocessing import Pool
from functools import partial

from new_utils import Consts, Dirs
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

    # INDEXES
    FILENAME = 0
    LABEL = 1

    if not os.path.exists(filepath):
        print("USER ERROR: invalid filepath value.")
        return

    identities = {}

    with open(filepath, 'r') as labels:
        for line in labels:
            line_split = line.split()

            identities[line_split[FILENAME]] = int(line_split[LABEL])-1

    return identities

def filter_identity_dict(identities, num_classes):
    """ Removes unnecessary keys from dictionary

    Arguments:
        identities {dictionary} -- {"filename": int(label)}
        num_classes {integer} -- number of classes allowed to remain in the dictionary

    Returns:
        dictionary -- version of identity_dict after removing unwanted entries
    """

    filtered_identity_dict = {}
    for key in identities.keys():
        if identities[key] < num_classes:
            filtered_identity_dict[key] = identities[key]

    return filtered_identity_dict

def labels_by_class_dict(labels_dict):
    """ In a sense, switches keys and values from labels_dict dictionary but since
    that way the new keys wouldn't be unique, values from repeated keys are appended to a list

    Arguments:
        label_dict {dictionary} -- {"filename": int(label)}

    Returns:
        dictionary -- {str(int(label)): [image_0, ..., image_n]}
    """

    labels_by_class = {}
    for key in labels_dict.keys():
        if labels_dict[key] not in labels_by_class.keys():
            labels_by_class[labels_dict[key]] = []

        labels_by_class[labels_dict[key]].append(key)

    return labels_by_class

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
        print("USER ERROR: invalid images_path value: {}".format(images_path))
        return

    if os.path.exists(output_dir):
        if os.listdir(output_dir):
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
        sub_dir = "{}{}/".format(images_path, str(labels_dict[file]))
        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)

        os.rename(images_path + file, sub_dir + file)

def batch_detect_and_crop(crop_dirs):
    """ Crops pictures into different margins around the detected face (yes, also detects faces)

    Arguments:
        crop_dirs {list} -- location for the no_crop folder (slash terminated), and all the different crop percentages (NO_CROP must be index 0)
    """

    # INDEXES
    NO_CROP = 0
    WITH_CROP = 1

    src_dir = crop_dirs[NO_CROP]
    images = os.listdir(src_dir)

    for crop_dir in crop_dirs:
        if crop_dir != src_dir:
            percentage = int(crop_dir.split("_")[-1][:-1])

            if not os.path.exists(crop_dir):
                os.mkdir(crop_dir)

            pool = Pool(8)
            func = partial(detect_and_crop, src_dir=src_dir, dst_dir=crop_dir, percentage=percentage)
            pool.map(func, images)
            pool.close()
            pool.join()

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
    # TODO: fazer o que o pylint manda na linha a seguir a esta
    for key in labels_dict_by_class.keys():
        if labels_dict_by_class[key]:
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

def generate_augmentation(filtered_dir, aug_mult):
    """ Traverses through the now structured directory of face pictures and generates augmented
    copies of every one

    Arguments:
        filtered_dir {string} -- base directory of organized face pictures
        aug_mult {integer} -- number of times each face pic will be augmented
    """

    print("NOW GENERATING AUGMENTED EXAMPLES...")

    transform = 0
    face_patch = 1

    for directory in os.listdir(filtered_dir):
        print("AUGMENTING {}".format(directory))

        source_dirs = [
            "{}{}/train/original/".format(filtered_dir, directory),
            "{}{}/val/original/".format(filtered_dir, directory)
        ]

        for sub_dir in source_dirs:

            for iden_dir in os.listdir(sub_dir):
                iden_dir_path = "{}{}/".format(sub_dir, iden_dir)

                for file_name in os.listdir(iden_dir_path):
                    file_name = "{}{}".format(iden_dir_path, file_name)

                    for i in range(aug_mult):
                        augment_image(file_name, transform, version=i)
                        augment_image(file_name, face_patch, version=i)

##############################################################
##############################################################

def main():
    if not os.path.exists(Dirs.STRUCTURED_DIR):
        os.mkdir(Dirs.STRUCTURED_DIR)

    cropped_dirs = Dirs.get_crop_dirs(cache=False)

    labels_dict = identity_dict("{}identity_CelebA.txt".format(Dirs.DATASET_DIR))

    # faz isto para ter so 500 classes no maximo
    filter_celeba_identities(Dirs.ORIGINAL_DIR, labels_dict, cropped_dirs[0], Consts.NUM_CLASSES)

    labels_dict = filter_identity_dict(labels_dict, Consts.NUM_CLASSES)

    # depois falta fazer crops as fotos (margin: 0, 5, 10, 15, 20, 25, 30)
    batch_detect_and_crop(cropped_dirs)

    for directory in cropped_dirs:
        # e depois falta dividir em train/val (90/10)
        train_val_split(directory, Consts.VAL_PERCENTAGE, labels_dict)

        # e so depois dividir em subpastas (uma pasta por iden)
        sub_dirs = [
            "{}train/original/".format(directory),
            "{}val/original/".format(directory)
        ]

        for sub_dir in sub_dirs:
            organize_folder(sub_dir, labels_dict)

    generate_augmentation(Dirs.STRUCTURED_DIR, Consts.AUG_MULT)

if __name__ == "__main__":
    main()
