from __future__ import absolute_import, division, print_function

import os

import numpy as np
import keras
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from load_celeba import load_image_filenames_and_labels
from generator import Generator
from utils import num_classes, checkpoint_path, cp_period, train_dir, train_dir_aug, raw_dir, log_dir, num_classes, multiprocessing, n_workers, batch_size, num_epochs, base_learning_rate, dropout_rate

def finetune():
    # train_images, train_labels, val_images, val_labels = load_image_filenames_and_labels()
    # train_batch_generator = Generator(train_images, train_labels, batch_size)
    # val_batch_generator = Generator(val_images, val_labels, batch_size)

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        shear_range=5,
        width_shift_range=0.10,
        height_shift_range=0.10,
    )

    val_datagen = ImageDataGenerator(
        rescale=1./255
    )

    train_batch_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size= (160, 160), # TODO
        batch_size= 64,  # TODO
        class_mode= "sparse",
        save_to_dir= train_dir_aug,
        save_format= "jpeg",
    )

    val_batch_generator = val_datagen.flow_from_directory(
        raw_dir +"val_split_" + str(num_classes) + "/",
        target_size=(160, 160), # TODO
        batch_size=64, # TODO
        class_mode='sparse',
    )

    base_model = load_model("facenet/model.h5")
    base_model.load_weights("facenet/weights.h5")
    base_model.trainable = False

    dropout_layer = keras.layers.Dropout(dropout_rate)

    prediction_layer = keras.layers.Dense(
        units=num_classes,
        activation=keras.activations.softmax
    )

    model = keras.Sequential([
        base_model,
        dropout_layer,
        prediction_layer,
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(
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
        steps_per_epoch=1, # TODO
        validation_steps=1 #TODO
    )

finetune()