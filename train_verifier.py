from __future__ import absolute_import, division, print_function

import os

import keras
from keras import layers

import numpy as np

from load_celeba import get_face_pairs
from model import binary_classification_svm
from euclidean_db import TRAIN, VAL
import utils

def train_verifier():
    train_vectors, train_labels = get_face_pairs(TRAIN)
    val_vectors, val_labels = get_face_pairs(VAL)

    val_data = (np.array(val_vectors), np.array(val_labels))
    
    model = binary_classification_svm()

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

    model.fit(
        np.array(train_vectors),
        np.array(train_labels),
        batch_size=utils.batch_size,
        epochs=utils.num_epochs,
        verbose=1,
        callbacks=[cp_callback, tensorboard],
        validation_data=val_data,
        shuffle=False, # i do that already
    )

    predictions = model.predict(np.array(val_vectors))

    i=0


train_verifier()