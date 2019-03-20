from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
from tensorflow import keras

from model import create_model
# from load_dataset import load_dataset
from load_celeba import load_dataset

checkpoint_dir = os.path.dirname(os.path.realpath(__file__))
checkpoint_path = checkpoint_dir + "/training_3/cp-{epoch:03d}.hdf5"
checkpoint_path = checkpoint_path.replace('\\', '/')

# (train_images, train_labels), (test_images, test_labels) = load_dataset("lfw-deepfunneled")
(train_images, train_labels), (test_images, test_labels) = load_dataset()

print("####")

num_classes = 10177

model = create_model(num_classes)

print("####")
print("####")

# Muda valor da epoch manualmente (checkpoint mais recente)
model.load_weights(checkpoint_path.format(epoch=50))

print("####")
print("####")
print("####")

# implement evaluate_generator
loss, acc = model.evaluate(test_images[:1000], test_labels[:1000])
print("Model accuracy: {:5.2f}%".format(100*acc))