from __future__ import absolute_import, division, print_function

import os

import keras
from keras import layers

from load_celeba import load_image_filenames_and_labels
from generator import Generator
from load_local_model import load_local_model
import utils

def finetune():
    """
        old way of diferentiating augmentation vs no augmentation.
        now it's done inside custom generator class
    """
    # # # if utils.AUGMENTATION:
    # # #     train_datagen = ImageDataGenerator(
    # # #         # rescale=1./255,
    # # #         # rotation_range=15,
    # # #         # shear_range=5,
    # # #         # width_shift_range=0.05,
    # # #         # height_shift_range=0.05,
    # # #         # horizontal_flip=True,
    # # #     )

    # # #     val_datagen = ImageDataGenerator(
    # # #         # rescale=1./255
    # # #     )

    # # #     train_batch_generator = train_datagen.flow_from_directory(
    # # #         utils.train_dir,
    # # #         target_size= (utils.image_width, utils.image_width),
    # # #         batch_size= utils.batch_size,
    # # #         class_mode= "sparse",
    # # #         # save_to_dir= utils.train_dir_aug,
    # # #         # save_format= "jpeg",
    # # #     )

    # # #     val_batch_generator = val_datagen.flow_from_directory(
    # # #         utils.val_dir,
    # # #         target_size=(utils.image_width, utils.image_width),
    # # #         batch_size=utils.batch_size,
    # # #         class_mode='sparse',
    # # #     )
    # # # else:
    # # #     train_images, train_labels, val_images, val_labels = load_image_filenames_and_labels()
    # # #     train_batch_generator = Generator(train_images, train_labels, utils.batch_size)
    # # #     val_batch_generator = Generator(val_images, val_labels, utils.batch_size)

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
            steps_per_epoch=(2000/utils.batch_size)+1, # TODO
            validation_steps=(110/utils.batch_size)+1 # TODO
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
    

finetune()