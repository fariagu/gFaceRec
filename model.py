from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
from tensorflow import keras


# Returns convnet model
def create_model(num_classes):
    model = tf.keras.models.Sequential([
        keras.layers.Conv2D(
            filters=32,
            kernel_size=(3,3),
            padding='same',
            activation=tf.keras.activations.relu,
            input_shape=(128, 128, 3)
        ),
        keras.layers.MaxPool2D(
            pool_size=(2,2)
        ),
        keras.layers.Conv2D(
            filters=128,
            kernel_size=(4,4),
            padding='same',
            activation=tf.keras.activations.relu
        ),
        keras.layers.MaxPool2D(
            pool_size=(2,2)
        ),
        keras.layers.Conv2D(
            filters=64,
            kernel_size=(4,4),
            padding='same',
            activation=tf.keras.activations.relu
        ),
        keras.layers.MaxPool2D(
            pool_size=(2,2)
        ),
        keras.layers.Conv2D(
            filters=32,
            kernel_size=(2,2),
            padding='same',
            activation=tf.keras.activations.relu
        ),
        keras.layers.Flatten(),
        # Feature Vector Layer
        keras.layers.Dense(
            units=128,
            activation=tf.keras.activations.relu
        ),
        keras.layers.Dropout(
            rate=0.2
        ),
        # Final Classification Layer
        keras.layers.Dense(
            units=num_classes,
            activation=tf.keras.activations.softmax
        )
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy']
    )

    return model