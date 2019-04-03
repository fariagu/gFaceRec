from __future__ import absolute_import, division, print_function

import os

import numpy as np
import keras
from keras import layers
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
import tensorflow_hub as hub

from load_celeba import load_image_filenames_and_labels
from generator import Generator
import utils


def load_facenet_model():
    base_model = load_model("facenet/model.h5")
    base_model.load_weights("facenet/weights.h5")
    base_model.trainable = False

    dropout_layer = keras.layers.Dropout(utils.dropout_rate)

    prediction_layer = keras.layers.Dense(
        units=utils.num_classes,
        activation=keras.activations.softmax
    )

    model = keras.Sequential([
        base_model,
        dropout_layer,
        prediction_layer,
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(
            lr=utils.base_learning_rate
        ),
        loss=keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy']
    )

    return model

def mn_feature_vector(x):
    fv_module = hub.Module(utils.mobilenet_feature_vector_url, trainable=True)
    return fv_module(x)

def load_mobilenet_model():
    IMAGE_SIZE = hub.get_expected_image_size(hub.Module(utils.mobilenet_feature_vector_url)) + [3]

    feature_vector_layer = layers.Lambda(mn_feature_vector, input_shape=IMAGE_SIZE)
    dropout_layer = keras.layers.Dropout(utils.dropout_rate)
    classification_layer = keras.layers.Dense(
        units=utils.num_classes,
        activation=keras.activations.softmax
    )
    model = keras.Sequential([
        feature_vector_layer,
        dropout_layer,
        classification_layer
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(
            lr=utils.base_learning_rate
        ),
        loss=keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy']
    )

    return model

def load_local_model():
    if utils.model_in_use == utils.FACENET:
        return load_facenet_model()
    else:
        return load_mobilenet_model()
        # se quiser retomar checkpoint:
        # model.load_weights("./mobilenet/model.hdf5")

def finetune():

    if utils.AUGMENTATION:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            # shear_range=5,
            width_shift_range=0.05,
            height_shift_range=0.05,
        )

        val_datagen = ImageDataGenerator(
            rescale=1./255
        )

        train_batch_generator = train_datagen.flow_from_directory(
            utils.train_dir,
            target_size= (utils.image_width, utils.image_width),
            batch_size= utils.batch_size,
            class_mode= "sparse",
            # save_to_dir= utils.train_dir_aug,
            # save_format= "jpeg",
        )

        val_batch_generator = val_datagen.flow_from_directory(
            utils.raw_dir +"val_split_" + str(utils.num_classes) + "/",
            target_size=(utils.image_width, utils.image_width),
            batch_size=utils.batch_size,
            class_mode='sparse',
        )
    else:
        train_images, train_labels, val_images, val_labels = load_image_filenames_and_labels()
        train_batch_generator = Generator(train_images, train_labels, utils.batch_size)
        val_batch_generator = Generator(val_images, val_labels, utils.batch_size)
    
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

    model.save_weights(utils.checkpoint_path.format(epoch=0))

    if utils.AUGMENTATION:
        model.fit_generator(
            generator=train_batch_generator,
            epochs=utils.num_epochs,
            callbacks=[cp_callback, tensorboard],
            verbose=1,
            validation_data=val_batch_generator,
            use_multiprocessing=utils.multiprocessing,
            workers=utils.n_workers,
            steps_per_epoch=1, # TODO
            validation_steps=1 #TODO
        )
    else:
        model.fit_generator(
            generator=train_batch_generator,
            epochs=utils.num_epochs,
            callbacks=[cp_callback, tensorboard],
            verbose=1,
            validation_data=val_batch_generator,
            use_multiprocessing=utils.multiprocessing,
            workers=utils.n_workers,
        )
    

# finetune()