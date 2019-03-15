from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
from tensorflow import keras

from model import create_model
from load_dataset import load_dataset
# from train import dataset_name, checkpoint_path

(train_images, train_labels), (test_images, test_labels) = load_dataset("lfw-deepfunneled")

print(len(test_labels))

num_classes = 500

model = create_model(num_classes)
model.load_weights("C:/code/keras/training_1/cp-0050.hdf5")

loss, acc = model.evaluate(test_images[:500], test_labels[:500])
print("Model accuracy: {:5.2f}%".format(100*acc))
