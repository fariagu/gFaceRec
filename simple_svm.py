from __future__ import absolute_import, division, print_function

import os

import keras
from keras import layers

import numpy as np

from load_celeba import load_vectors
from model import vgg_svm
from euclidean_db import TRAIN, VAL
from vector_generator import VectorGenerator
import utils

def svm():
    vector_paths, labels = load_vectors()
    train_split_index = int(len(labels)*0.8)
    
    train_generator = VectorGenerator(
        vector_paths[:train_split_index],
        labels[:train_split_index],
        utils.batch_size
    )

    val_generator = VectorGenerator(
        vector_paths[train_split_index:],
        labels[train_split_index:],
        utils.batch_size
    )

    model = vgg_svm()

    model.summary()
    tensorboard = keras.callbacks.TensorBoard(log_dir=utils.log_dir)

    # Load Checkpoints
    cp_callback = keras.callbacks.ModelCheckpoint(
        utils.checkpoint_path,
        verbose=1,
        save_weights_only=False,
        period=utils.cp_period
    )

    model.save_weights(utils.checkpoint_path.format(epoch=0))

    model.fit_generator(
        generator=train_generator,
        epochs=utils.num_epochs,
        callbacks=[cp_callback, tensorboard],
        verbose=1,
        validation_data=val_generator,
        use_multiprocessing=utils.multiprocessing,
        workers=utils.n_workers,
        shuffle=False, # I do that already
    )

    # predictions = model.predict(np.array(val_vectors))

svm()