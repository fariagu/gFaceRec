from __future__ import absolute_import, division, print_function

import os
import concurrent.futures

from PIL import Image
import numpy as np

size = 128, 128

raw_dir = "C:/datasets/"
cache_dir = "C:/dataset_cache/"

def load_image(file_path):
    img = Image.open(file_path)
    img.thumbnail(size)
    img.load()
    data = np.asarray(img, dtype="uint8")
    return data

def write_string_labels(dataset_name, str_labels):
    path = cache_dir + dataset_name +  "/labels.txt"
    with open(path, 'w') as f:
        for i, label in enumerate(str_labels):
            f.write(str(i) + " " + label)

def load_raw_dataset(dataset_name):
    train_images = []
    test_images = []
    train_labels = []
    test_labels = []
    str_labels = []

    dataset_path = raw_dir + dataset_name + "/"

    for subdir, dirs, files in os.walk(dataset_path):
        label = subdir.split("/")[-1]
        
        for i, file in enumerate(files):
            file_path = subdir + "/" + file

            if i == 0 and len(files) > 1:
                test_images.append(load_image(file_path))
                test_labels.append(i)
                str_labels.append(label)
            else:
                train_images.append(load_image(file_path))
                train_labels.append(i)
            
        
    np_train_images = np.array(train_images)
    np_train_labels = np.array(train_labels)
    np_test_images = np.array(test_images)
    np_test_labels = np.array(test_labels)

    dataset_cache_dir = cache_dir + dataset_name + "/"

    train_images_cache = dataset_cache_dir + dataset_name + "_train_images"
    train_labels_cache = dataset_cache_dir + dataset_name + "_train_labels"
    test_images_cache = dataset_cache_dir + dataset_name + "_test_images"
    test_labels_cache = dataset_cache_dir + dataset_name + "_test_labels"
    np.savez_compressed(train_images_cache, np_train_images)
    np.savez_compressed(train_labels_cache, np_train_labels)
    np.savez_compressed(test_images_cache, np_test_images)
    np.savez_compressed(test_labels_cache, np_test_labels)
    write_string_labels(dataset_name, str_labels)


    return (np_train_images, np_train_labels), (np_test_images, np_test_labels)

def load_npy_dataset(dataset_name):
    train_images_path = cache_dir + dataset_name + "/" + dataset_name + "_train_images.npz"
    train_labels_path = cache_dir + dataset_name + "/" + dataset_name + "_train_labels.npz"
    test_images_path = cache_dir + dataset_name + "/" + dataset_name + "_test_images.npz"
    test_labels_path = cache_dir + dataset_name + "/" + dataset_name + "_test_labels.npz"
    train_images_npz = np.load(train_images_path)
    train_labels_npz = np.load(train_labels_path)
    test_images_npz = np.load(test_images_path)
    test_labels_npz = np.load(test_labels_path)

    train_images = train_images_npz[train_images_npz.files[0]]
    train_labels = train_labels_npz[train_labels_npz.files[0]]
    test_images = test_images_npz[test_images_npz.files[0]]
    test_labels = test_labels_npz[test_labels_npz.files[0]]
    
    return (train_images, train_labels), (test_images, test_labels)

def load_dataset(dataset_name):
    dataset_cache_dir = cache_dir + dataset_name + "/"

    if not os.path.exists(dataset_cache_dir):
        os.makedirs(dataset_cache_dir)
        return load_raw_dataset(dataset_name)
    else:
        return load_npy_dataset(dataset_name)
    
def main():
    (train_images, train_labels), (test_images, test_labels) = load_dataset("lfw-deepfunneled")
    print(len(train_labels))

    # print(train_images.shape)
    # print(train_labels.shape)
    # print(test_images.shape)
    # print(test_labels.shape)