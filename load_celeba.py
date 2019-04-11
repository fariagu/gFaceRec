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
from euclidean_db import predict, TRAIN, VAL
from generator import Generator
from load_local_model import load_facenet_fv

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
            if int(line_split[LABEL]) < 10:
                identity_dict[line_split[FILENAME]] = int(line_split[LABEL])-1
    
    pickle.dump(identity_dict, open(utils.identity_cache_dir, "wb"))

    return identity_dict

# every image (no splits)
def filenames_and_labels():
    if os.path.exists(utils.identity_cache_dir):
        return pickle.load(open(utils.identity_cache_dir, "rb"))
    else:
        return filenames_and_labels_from_disk()

def prepend_string_to_array(string, array):
    res = []
    for elem in array:
        res.append(string + elem)
    
    return res

def load_vectors_into_disk():
    identity_dict = filenames_and_labels()
    file_names = list(identity_dict.keys())
    labels = list(identity_dict.values())
    paths = prepend_string_to_array(utils.images_dir, file_names)
    full_generator = Generator(paths, labels, utils.batch_size)

    model = load_facenet_fv()

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

def load_vectors():
    identity_dict = filenames_and_labels()
    vectors = []

    if os.path.exists(utils.vector_dir):
        for vector in os.listdir(utils.vector_dir):
            index = vector.split(".")[0] + ".jpg"
            vectors.append((pickle.load(open(utils.vector_dir + vector, "rb")), identity_dict[index]))
            # file
            pass
    
    return vectors

def generate_image_pairs(split_str):
    train_data, train_data_mean, train_data_std = predict(split_str)

    face_pairs = []
    vectors = []

    for person in train_data[0].items():
        identity = person[0]
        face_vectors = person[1]

        for i, vector in enumerate(face_vectors):
            if i + 2 < len(face_vectors) and i < 30:
                two_in_one = np.concatenate((face_vectors[i], face_vectors[i+1]), axis=None)
                face_pairs.append((two_in_one, SAME))
            
            vectors.append((vector, identity))

    # shuffle 
    rand.shuffle(vectors)

    for i in range(0, len(vectors)-2):
        two_in_one = np.concatenate((vectors[i][VECTOR], vectors[i+1][VECTOR]), axis=None)
        face_pairs.append((
            two_in_one,
            SAME if vectors[i][LABEL] == vectors[i+1][LABEL] else DIFF
        ))
        i += 2

    rand.shuffle(face_pairs)

    # # just for testing
    # pairs_same = 0
    # pairs_diff = 0
    # for pair in face_pairs:
    #     if pair[1] == SAME:
    #         pairs_same += 1
    #     else:
    #         pairs_diff += 1


    pickle.dump(face_pairs, open(utils.cache_dir + "face_pairs_" + split_str + "_" + str(utils.num_classes) + ".pkl", "wb"))

    return zip(*face_pairs)

def get_face_pairs(split_str):
    path = utils.cache_dir + "face_pairs_" + split_str + "_" + str(utils.num_classes) + ".pkl"
    if os.path.exists(path):
        return zip(*pickle.load(open(path, "rb")))

    return generate_image_pairs(split_str)


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

# load_train_val_test_from_txt()
# get_face_pairs(TRAIN)
# get_face_pairs(VAL)
# load_test_data_from_txt()
load_vectors_into_disk()
# load_vectors()
# filenames_and_labels_from_disk()