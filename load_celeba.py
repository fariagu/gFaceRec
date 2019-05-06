from __future__ import absolute_import, division, print_function

import os
# import concurrent.futures

import random as rand
# import gc
# from PIL import Image
import numpy as np
import pickle
from keras.models import load_model

import utils
# from euclidean_db import predict, TRAIN, VAL
from generator import Generator
from load_local_model import load_facenet_fv, load_vgg_face_fv

SAME = 0
DIFF = 1

FILENAME = 0
VECTOR = 0
LABEL = 1

"""
    nao faz muito porque neste dataset so tenho labels numericas
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

def load_train_val_test_from_txt():

    train_val_test = {}
    for file in os.listdir(utils.images_dir):
        # 80% train, 10% validation, 10% test
        seed = rand.randint(1, 20)
        if seed == 20:
            train_val_test[file] = "2"
        elif seed == 19:
            train_val_test[file] = "1"
        else:
            train_val_test[file] = "0"

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
            if int(label) < utils.num_classes and file_name in train_val_test:

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

def filenames_and_labels_from_disk():
    identity_dict = {}

    with open(utils.labels_path, 'r') as f:
        for line in f:
            line_split = line.split()

            # temporary (while testing)
            if int(line_split[LABEL]) < utils.num_classes:
                identity_dict[line_split[FILENAME]] = int(line_split[LABEL])-1
    
    pickle.dump(identity_dict, open(utils.identity_cache_dir, "wb"))

    return identity_dict

# every image (no splits)
def filenames_and_labels():
    if os.path.exists(utils.identity_cache_dir):
        with open(utils.identity_cache_dir, "rb") as f:
            return pickle.load(f)
    else:
        return filenames_and_labels_from_disk()

def prepend_string_to_array(string, array):
    res = []
    for elem in array:
        res.append(string + elem)
    
    return res

def load_vectors_into_disk():
    identity_dict = filenames_and_labels()

    print("####################")
    print(utils.images_dir)

    # trim_iden_dict = {}
    # for key, label in identity_dict.items():
    #     if label < utils.num_classes:
    #         trim_iden_dict[key] = label
    #
    # file_names = list(trim_iden_dict.keys())
    # labels = list(trim_iden_dict.values())

    files_that_exist = []
    final_dict = {}
    for file in os.listdir(utils.raw_dir + "img_crop_25_aug_times_5/"):
    # for file in os.listdir(utils.images_dir):
        files_that_exist.append(file.split("_")[-1])
        final_dict[file] = -1

    for file in list(identity_dict.keys()):
        if file not in files_that_exist:
            identity_dict.pop(file)

    print(len(identity_dict))

    
    for file in list(final_dict.keys()):
        if file.split("_")[-1] in identity_dict.keys():
            final_dict[file] = identity_dict[file.split("_")[-1]]
            pass

    file_names = list(final_dict.keys())
    labels = list(final_dict.values())

    # paths = prepend_string_to_array(utils.images_dir, file_names)
    paths = prepend_string_to_array(utils.raw_dir + "img_crop_25_aug_times_5/", file_names)
    full_generator = Generator(paths, labels, utils.batch_size)

    model = load_facenet_fv() if utils.model_in_use == utils.FACENET else load_vgg_face_fv()

    model.summary()

    predictions = model.predict_generator(
        generator=full_generator,
        verbose=1,
        use_multiprocessing=utils.multiprocessing,
        workers=utils.n_workers,
    )

    if not os.path.exists(utils.vector_dir):
        os.mkdir(utils.vector_dir)

    for fv, path in zip(predictions, paths):
        file_name = path.split("/")[-1].split(".")[0]
        pickle.dump(fv, open(utils.vector_dir + file_name + ".pkl", "wb"))

# chamar isto para carregar os vectores em memória (num dict)
def load_vec_dict():
    identity_dict = filenames_and_labels()
    vectors = []

    for vector_file in os.listdir(utils.vector_dir):
        vectors.append(vector_file)
    
    rand.shuffle(vectors)

    split_index = int(len(vectors)*0.8)
    train_split = vectors[:split_index]
    val_split = vectors[split_index:]

    train_v_dict = {}
    val_v_dict = {}

    for t_vector in train_split:
        iden_key = t_vector.split(".")[0] + ".jpg"
        label = str(identity_dict[iden_key])
        if label not in train_v_dict:
            train_v_dict[label] = []
        
        with open(utils.vector_dir + t_vector, "rb") as v:
            vector = pickle.load(v)
            train_v_dict[label].append(vector)
    
    for v_vector in val_split:
        iden_key = v_vector.split(".")[0] + ".jpg"
        label = str(identity_dict[iden_key])
        if label not in val_v_dict:
            val_v_dict[label] = []
        
        with open(utils.vector_dir + v_vector, "rb") as v:
            vector = pickle.load(v)
            val_v_dict[label].append(vector)
    
    return train_v_dict, val_v_dict

# chamar isto para ter a info a passar ao vector_generator
def load_vectors():
    identity_dict = filenames_and_labels()
    vector_paths = []
    labels = []

    if os.path.exists(utils.vector_dir):
        for vector in os.listdir(utils.vector_dir):
            index = vector.split(".")[0] + ".jpg"
            if identity_dict[index] < utils.num_classes:
                vector_paths.append(utils.vector_dir + vector)
                labels.append(identity_dict[index])
        
    aggregated_vl = list(zip(vector_paths, labels))
    rand.shuffle(aggregated_vl)

    return zip(*aggregated_vl)

def create_pairs():
    # MACROS
    VECTOR = 0
    LABEL = 1
    SAME = 0
    DIFF = 1

    pairs = []
    vector_paths, labels = load_vectors()
    it = list(range(len(labels)))
    rand.shuffle(it)
    agg = list(zip(vector_paths, labels))

    # random pairing (will be mostly pairs of different faces)
    for i in it:
        face1 = agg[i]
        for j in range(i+1, i+5):
            # for when j is out of range
            j = j - len(labels) if j >= len(labels) else j

            face2 = agg[j]

            binary_label = SAME if face1[LABEL] == face2[LABEL] else DIFF
            pairs.append(((face1[VECTOR], face2[VECTOR]), binary_label))
    
    #specific pairing
    # sort by label
    agg.sort(key=lambda tuple: tuple[LABEL])

    for i in range(len(labels)-1):
        face1 = agg[i]
        for j in range(i+1, i+3):
            j = j - len(labels) if j >= len(labels) else j
            face2 = agg[j]

            binary_label = SAME if face1[LABEL] == face2[LABEL] else DIFF
            pairs.append(((face1[VECTOR], face2[VECTOR]), binary_label))
        
        i += 1
    
    rand.shuffle(pairs)

    print("#######################")
    same_face = 0
    diff_face = 0
    for pair in pairs:
        if pair[1] == SAME:
            same_face += 1
        else:
            diff_face += 1

    print("Pairs of same face: " + str(same_face))
    print("Pairs of diff face: " + str(diff_face))
    print("#######################")


    return zip(*pairs)

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
            if int(label) < utils.num_classes and file_name in train_val_test:

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

load_vectors_into_disk()