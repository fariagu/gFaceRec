from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow_hub as hub

import keras
from keras import layers
from keras.preprocessing.image import ImageDataGenerator

from load_celeba import load_image_filenames_and_labels
from generator import Generator
from utils import raw_dir, num_classes, batch_size, log_dir, checkpoint_path, base_learning_rate, cp_period, multiprocessing, n_workers, num_epochs, training_session, dropout_rate

fv_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/2"


def feature_vector(x):
    fv_module = hub.Module(fv_url, trainable=True)
    return fv_module(x)

def load_module_as_model():
    IMAGE_SIZE = hub.get_expected_image_size(hub.Module(fv_url)) + [3]

    feature_vector_layer = layers.Lambda(feature_vector, input_shape=IMAGE_SIZE)
    noise_layer = layers.GaussianNoise(0.1)
    dropout_layer = keras.layers.Dropout(dropout_rate)
    classification_layer = keras.layers.Dense(
        units=num_classes,
        activation=keras.activations.softmax
    )
    model = keras.Sequential([
        feature_vector_layer,
        noise_layer,
        dropout_layer,
        classification_layer
    ])

    model.compile(
        optimizer=keras.optimizers.RMSprop(
            lr=base_learning_rate
        ),
        loss=keras.losses.hinge,
        metrics=['accuracy']
    )

    return model

def ftmobilenet():
    # train_images, train_labels, val_images, val_labels = load_image_filenames_and_labels()
    
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

    # Not augmented
    # train_batch_generator = Generator(train_images, train_labels, batch_size)

    val_batch_generator = val_datagen.flow_from_directory(
        raw_dir +"val_split_100/",
        target_size=(224, 224), # TODO
        batch_size=32, # TODO
        class_mode='sparse'
    )

    # val_batch_generator = Generator(val_images, val_labels, batch_size)

    model = load_module_as_model()
    # model.load_weights("./mobilenet/model.hdf5")
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
        validation_steps=104/4 #TODO
    )

    # model.save_weights("./mobilenet/training_{training:04d}/weights.hdf5".format(training=training_session))

ftmobilenet()