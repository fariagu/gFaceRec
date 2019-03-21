from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
from tensorflow import keras

from model import create_model
from load_celeba import load_image_filenames_and_labels
from generator import Generator
from utils import batch_size, log_dir, checkpoint_path, training_session, cp_period, num_epochs

def main():

    train_images, train_labels, val_images, val_labels = load_image_filenames_and_labels()
    
    train_batch_generator = Generator(train_images, train_labels, batch_size)
    val_batch_generator = Generator(val_images, val_labels, batch_size)

    # Load Checkpoints
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        verbose=1,
        save_weights_only=False,
        period=cp_period
    )
    
    tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir)

    model = create_model()
    model.summary()
    model.save_weights(checkpoint_path.format(epoch=0))

    model.fit_generator(
        generator=train_batch_generator,
        epochs=num_epochs,
        callbacks=[cp_callback, tensorboard],
        verbose=1,
        validation_data=val_batch_generator,
        use_multiprocessing=True,
    )

main()