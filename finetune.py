from __future__ import absolute_import, division, print_function

import os

import numpy as np
import keras
from keras.models import load_model

from load_celeba import load_image_filenames_and_labels
from generator import Generator
from utils import num_classes, checkpoint_path, cp_period, log_dir, num_classes, multiprocessing, n_workers, batch_size, num_epochs, base_learning_rate

def finetune():
    train_images, train_labels, val_images, val_labels = load_image_filenames_and_labels()
        
    train_batch_generator = Generator(train_images, train_labels, batch_size)
    val_batch_generator = Generator(val_images, val_labels, batch_size)

    base_model = load_model("facenet/model.h5")
    base_model.load_weights("facenet/weights.h5")
    base_model.trainable = False

    prediction_layer = keras.layers.Dense(
        units=num_classes,
        activation=keras.activations.softmax
    )

    model = keras.Sequential([
        base_model,
        prediction_layer,
    ])

    model.compile(
        optimizer=keras.optimizers.RMSprop(
            lr=base_learning_rate
        ),
        loss=keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy']
    )

    model.summary()
    tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir)

    # Load Checkpoints
    cp_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        verbose=1,
        save_weights_only=False,
        period=cp_period
    )

    model.save_weights(checkpoint_path.format(epoch=0))

    model.fit_generator(
        generator=train_batch_generator,
        epochs=num_epochs,
        callbacks=[cp_callback, tensorboard],
        verbose=1,
        validation_data=val_batch_generator,
        use_multiprocessing=multiprocessing,
        workers=n_workers,
    )

finetune()