from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
from tensorflow import keras

from model import create_model
# from load_dataset import load_dataset
from load_celeba import load_dataset, get_num_classes

# dataset_name = "test-lfw"
# dataset_name = "lfw-deepfunneled"

#print(tf.__version__)

(train_images, train_labels), (test_images, test_labels) = load_dataset()

# use only first 1000 examples
train_images = train_images[:1000]
train_labels = train_labels[:1000]
test_images = test_images[:1000]
test_labels = test_labels[:1000]

checkpoint_dir = os.path.dirname(os.path.realpath(__file__))
checkpoint_path = checkpoint_dir + "/training_3/cp-{epoch:03d}.hdf5"
checkpoint_path = checkpoint_path.replace('\\', '/')

# Load Checkpoints
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    verbose=1,
    save_weights_only=False,
    # save weights every 5 epochs
    period=10
)
tensorboard = keras.callbacks.TensorBoard(log_dir="./logs")

model = create_model(get_num_classes())
model.summary()
model.save_weights(checkpoint_path.format(epoch=0))

# TODO: calcular epochs tendo em conta o training size
model.fit(
    train_images,
    train_labels,
    epochs=50,
    validation_data=(test_images, test_labels),
    callbacks=[cp_callback, tensorboard],
    verbose=0
)