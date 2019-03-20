from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
from tensorflow import keras

from model import create_model
# from load_dataset import load_dataset
# from load_celeba import load_dataset, get_num_classes
from load_celeba import load_image_filenames_and_labels
from generator import Generator

# dataset_name = "test-lfw"
# dataset_name = "lfw-deepfunneled"

#print(tf.__version__)

def main():
    # (train_images, train_labels), (test_images, test_labels) = load_dataset()

    # use only first 1000 examples
    # train_images = train_images[:1000]
    # train_labels = train_labels[:1000]
    # test_images = test_images[:1000]
    # test_labels = test_labels[:1000]

    train_images, train_labels, val_images, val_labels, test_images, test_labels = load_image_filenames_and_labels()
    batch_size = 8
    train_batch_generator = Generator(train_images, train_labels, batch_size)
    val_batch_generator = Generator(val_images, val_labels, batch_size)

    # print(train_batch_generator.shape)

    checkpoint_dir = os.path.dirname(os.path.realpath(__file__))
    checkpoint_path = checkpoint_dir + "/training_5/cp-{epoch:04d}.hdf5"
    checkpoint_path = checkpoint_path.replace('\\', '/')

    # Load Checkpoints
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        verbose=1,
        save_weights_only=False,
        # save weights every N epochs
        period=5
    )
    tensorboard = keras.callbacks.TensorBoard(log_dir="./logs")

    # model = create_model(get_num_classes())
    model = create_model(100)  # 1000 classes
    model.summary()
    model.save_weights(checkpoint_path.format(epoch=0))

    # TODO: calcular epochs tendo em conta o training size
    # model.fit(
    #     train_images,
    #     train_labels,
    #     epochs=50,
    #     validation_data=(test_images, test_labels),
    #     callbacks=[cp_callback, tensorboard],
    #     verbose=0
    # )

    model.fit_generator(
        generator=train_batch_generator,
        epochs=100,
        callbacks=[cp_callback, tensorboard],
        verbose=1,
        validation_data=val_batch_generator,
    )

main()