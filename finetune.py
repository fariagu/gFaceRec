from __future__ import absolute_import, division, print_function

import os

import numpy as np
import keras
from keras.models import load_model

from load_celeba import load_image_filenames_and_labels
from generator import Generator
from utils import num_classes, checkpoint_path, cp_period, log_dir, num_classes, multiprocessing, n_workers, batch_size, num_epochs, base_learning_rate

def finetune():
    # train_images, train_labels, val_images, val_labels = load_image_filenames_and_labels()
        
    # train_batch_generator = Generator(train_images, train_labels, batch_size)
    # val_batch_generator = Generator(val_images, val_labels, batch_size)

    train_datagen = ImageDataGenerator(
        # rescale=1./255
        # featurewise_center=True,
        rotation_range=5,
        shear_range=1.0,
        width_shift_range=0.01,
        height_shift_range=0.01,
    )

    val_datagen = ImageDataGenerator(
        # rescale=1./255
    )

    # Augmented
    train_batch_generator = train_datagen.flow_from_directory(
        raw_dir +"train_split_100/",
        target_size=(224, 224), # TODO
        batch_size=32,  # TODO
        class_mode='sparse'
    )

    val_batch_generator = val_datagen.flow_from_directory(
        raw_dir +"val_split_100/",
        target_size=(224, 224), # TODO
        batch_size=32, # TODO
        class_mode='sparse'
    )

    base_model = load_model("facenet/model.h5")
    base_model.load_weights("facenet/weights.h5")
    base_model.trainable = False

    dropout_layer = keras.layers.Dropout(0.5)

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
        steps_per_epoch=2025/batch_size, # TODO
        validation_steps=105/4 #TODO
    )

finetune()