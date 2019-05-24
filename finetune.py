from __future__ import absolute_import, division, print_function

import os

import keras
from keras import layers

# from load_celeba import load_image_filenames_and_labels
from load_dataset import load_image_filenames_and_labels
from generator import Generator
from load_local_model import load_local_model
import utils

def finetune(epochs):
    # train_images, train_labels, val_images, val_labels = load_image_filenames_and_labels()
    train_images, train_labels, val_images, val_labels = load_image_filenames_and_labels(
        
    )
    train_batch_generator = Generator(
        train_images,
        train_labels,
        utils.batch_size,
        save_to_dir=utils.SAVE_TO_DIR
    )
    val_batch_generator = Generator(
        val_images,
        val_labels,
        utils.batch_size,
        save_to_dir=utils.SAVE_TO_DIR
    )
    
    model = load_local_model()

    model.summary()
    tensorboard = keras.callbacks.TensorBoard(log_dir=utils.log_dir)

    # Load Checkpoints
    cp_callback = keras.callbacks.ModelCheckpoint(
        utils.checkpoint_path,
        verbose=1,
        save_weights_only=False,
        period=utils.cp_period
    )

    # model.save_weights(utils.checkpoint_path.format(epoch=0))

    model.fit_generator(
        generator=train_batch_generator,
        epochs=epochs,
        callbacks=[cp_callback, tensorboard],
        verbose=1,
        validation_data=val_batch_generator,
        use_multiprocessing=utils.multiprocessing,
        workers=utils.n_workers,
    )
    
# finetune(utils.num_epochs)