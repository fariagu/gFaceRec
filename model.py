from __future__ import absolute_import, division, print_function

import os

import keras

# Returns convnet model
def create_model(num_classes):
    model = keras.models.Sequential([
        keras.layers.Conv2D(
            filters=32,
            kernel_size=(4,4),
            padding='same',
            activation=keras.activations.relu,
            input_shape=(128, 128, 3)
        ),
        keras.layers.MaxPool2D(
            pool_size=(2,2)
        ),
        keras.layers.Dropout(
            rate=0.25
        ),
        keras.layers.Conv2D(
            filters=16,
            kernel_size=(4,4),
            padding='same',
            activation=keras.activations.relu
        ),
        keras.layers.MaxPool2D(
            pool_size=(2,2)
        ),
        keras.layers.Dropout(
            rate=0.25
        ),
        keras.layers.Conv2D(
            filters=8,
            kernel_size=(4,4),
            padding='same',
            activation=keras.activations.relu
        ),
        keras.layers.MaxPool2D(
            pool_size=(2,2)
        ),
        keras.layers.Dropout(
            rate=0.25
        ),
        keras.layers.Conv2D(
            filters=4,
            kernel_size=(2,2),
            padding='same',
            activation=keras.activations.relu
        ),
        keras.layers.Dropout(
            rate=0.25
        ),
        keras.layers.Flatten(),
        # Feature Vector Layer
        keras.layers.Dense(
            units=128,
            activation=keras.activations.relu
        ),
        keras.layers.Dropout(
            rate=0.25
        ),
        # Final Classification Layer
        keras.layers.Dense(
            units=num_classes,
            activation=keras.activations.softmax
        )
    ])

    model.compile(
        # optimizer=keras.optimizers.Adam(),
        optimizer=keras.optimizers.RMSprop(),
        loss=keras.losses.sparse_categorical_crossentropy,
        metrics=[
            'accuracy',
        ]
    )

    return model