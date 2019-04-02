from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
import keras
from keras.models import load_model

from keras.applications.inception_v3 import InceptionV3

from model import create_model
from load_celeba import load_image_filenames_and_labels
from generator import Generator
import utils

def main():

    train_images, train_labels, val_images, val_labels = load_image_filenames_and_labels()
    
    train_batch_generator = Generator(train_images, train_labels, utils.batch_size)
    val_batch_generator = Generator(val_images, val_labels, utils.batch_size)

    # Load Checkpoints
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        utils.checkpoint_path,
        verbose=1,
        save_weights_only=False,
        period=utils.cp_period
    )
    
    tensorboard = keras.callbacks.TensorBoard(log_dir=utils.log_dir)

    model = create_model()
    model.summary()
    model.save_weights(utils.checkpoint_path.format(epoch=0))

    model.fit_generator(
        generator=train_batch_generator,
        epochs=utils.num_epochs,
        callbacks=[cp_callback, tensorboard],
        verbose=1,
        validation_data=val_batch_generator,
        use_multiprocessing=utils.multiprocessing,
        workers=utils.n_workers,
    )

    
# main()

# base_model = load_model("facenet/model.h5")
# base_model.layers.pop()
# base_model.layers.pop()
# base_model.layers.pop()
# base_model.layers.pop()
# base_model = InceptionV3(include_top=False)
# base_model.summary()