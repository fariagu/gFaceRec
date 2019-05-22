from __future__ import absolute_import, division, print_function

import keras

from load_dataset_cache import vectors_and_labels
from vector_generator import VectorGenerator
from load_local_model import classifier

from new_utils import Consts, Dirs, Flags
from params import Params, HyperParams, Config

def read_training_session():
    read_file = open("training_session.txt", "r")
    training_session = int(read_file.read())
    print("Training Session {}".format(training_session))
    read_file.close()
    write_file = open("training_session.txt", "w")
    write_file.write(str(training_session+1))
    write_file.close()

    return training_session

def get_fv_len(model):
    if model not in Consts.MODELS:
        print("USER ERROR: invalid model value.")
        return -1

    if model == Consts.INCEPTIONV3:
        return 128

    return 512

def train_classifier(params, hyper_params, train_config, val_config):
    train_paths, train_labels = vectors_and_labels(params, train_config)
    val_paths, val_labels = vectors_and_labels(params, val_config)
    cp_period = hyper_params.num_epochs / 10

    training_session = read_training_session()

    train_generator = VectorGenerator(
        train_paths,
        train_labels,
        hyper_params.batch_size # Consts.BATCH_SIZE
    )

    val_generator = VectorGenerator(
        val_paths,
        val_labels,
        hyper_params.batch_size # Consts.BATCH_SIZE
    )

    model = classifier(
        fv_len=get_fv_len(params.model),
        num_classes=params.num_classes,
        dropout_rate=hyper_params.dropout_rate,
        final_layer_activation=keras.activations.softmax,
        optimizer=keras.optimizers.rmsprop
    )

    model.summary()
    tensorboard = keras.callbacks.TensorBoard(
        log_dir=Dirs.LOG_DIR(sess=training_session)
    )

    # Load Checkpoints
    cp_callback = keras.callbacks.ModelCheckpoint(
        Dirs.CHECKPOINT_PATH,
        verbose=1,
        save_weights_only=False,
        period=cp_period
    )

    model.save_weights(Dirs.CHECKPOINT_PATH.format(sess=training_session, epoch=0))

    model.fit_generator(
        generator=train_generator,
        epochs=hyper_params.num_epochs,
        callbacks=[cp_callback, tensorboard],
        verbose=1,
        validation_data=val_generator,
        use_multiprocessing=not Flags.ENV_WINDOWS,
        workers=Consts.N_WORKERS,
        shuffle=False, # I do that already
    )
