from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
from tensorflow import keras

from model import create_model
from load_dataset import load_dataset

# dataset_name = "test-lfw"
dataset_name = "lfw-deepfunneled"

#print(tf.__version__)

# TODO: do this locally for celeb_a dataset (ou vggface2 ( 30+GB :( ))
(train_images, train_labels), (test_images, test_labels) = load_dataset(dataset_name)

print(len(train_labels))


# use only first 500 examples
# shape = (-1, 28, 28[, 1]) -> do meu vai ser (-1, 160, 160, 3)
train_images = train_images[:500]
train_labels = train_labels[:500]
test_images = test_images[:500]
test_labels = test_labels[:500]

checkpoint_dir = os.path.dirname(os.path.realpath(__file__))
checkpoint_path = checkpoint_dir + "/training_1/cp-{epoch:04d}.hdf5"
checkpoint_path = checkpoint_path.replace('\\', '/')

# Load Checkpoints
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    verbose=1,
    save_weights_only=False,
    # save weights every 5 epochs
    period=5
)

model = create_model(len(train_labels))
model.summary()
model.save_weights(checkpoint_path.format(epoch=0))

# TODO: calcular epochs tendo em conta o training size
model.fit(
    train_images,
    train_labels,
    epochs=50,
    validation_data=(test_images, test_labels),
    callbacks=[cp_callback],
    verbose=0
)