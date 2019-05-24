from __future__ import absolute_import, division, print_function

import os

import keras

from load_dataset_cache import vectors_and_labels
from vector_generator import VectorGenerator
from load_local_model import classifier

from new_utils import Consts, Dirs, Flags
from params import Params, HyperParams, Config

def init_log_dir():
    with open(Dirs.TRAINING_SESSION_PATH, "w") as write_file:
        write_file.write("1")
        write_file.close()

def read_training_session():
    if not os.path.exists(Dirs.LOG_BASE_DIR):
        os.mkdir(Dirs.LOG_BASE_DIR)
        init_log_dir()

    if not os.path.exists(Dirs.TRAINING_BASE_DIR):
        os.mkdir(Dirs.TRAINING_BASE_DIR)

    training_session = -1

    with open(Dirs.TRAINING_SESSION_PATH, "r") as read_file:
        training_session = int(read_file.read())
        print("Training Session {}".format(training_session))
        read_file.close()

    with open(Dirs.TRAINING_SESSION_PATH, "w") as write_file:
        write_file.write(str(training_session+1))
        write_file.close()

    checkpoint_dir = Dirs.CHECKPOINT_BASE_DIR.format(sess=training_session)
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    return training_session

def save_session_params(params, hyper_params, train_config, val_config):
    training_session = read_training_session()

    if not os.path.exists(Dirs.LOG_DIR.format(sess=training_session)):
        os.mkdir(Dirs.LOG_DIR.format(sess=training_session))

    configs = [train_config, val_config]
    with open(Dirs.PARAMS_PATH.format(sess=training_session), "w") as file:
        file.write("Session: " + str(training_session) + "\n\n")
        file.write("Params:\n")
        file.write("\tModel: " + params.model + "\n")
        file.write("\tNumber of classes: " + str(params.num_classes) + "\n")
        file.write("\tExamples per class: " + str(params.examples_per_class) + "\n")
        file.write("\tExamples per class: " + str(params.examples_per_class) + "\n")
        file.write("\tCrop percentage: " + str(params.crop_pctg) + "\n")
        file.write("\tInclude unknown:" + str(params.crop_pctg) + "\n")

        for config in configs:
            file.write("\n" + config.split.capitalize() + " Config:\n")
            for version in config.list_versions:
                file.write("\tInclude" + version[1] + ": " + str(version[0]) + "\n")

        file.write("\nHyperparams:\n")
        file.write("\tNumber of epochs: " + str(hyper_params.num_epochs) + "\n")
        file.write("\tDropout rate: " + str(hyper_params.dropout_rate) + "\n")
        file.write("\tBatch size: " + str(hyper_params.batch_size) + "\n")
        file.write("\tFinal layer activation: " + str(hyper_params.final_layer_activation) + "\n")
        file.write("\tOptimizer: " + str(hyper_params.optimizer) + "\n")

        file.close()

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

    training_session = save_session_params(params, hyper_params, train_config, val_config)
    checkpoint_path = Dirs.CHECKPOINT_BASE_DIR.format(sess=training_session) + "cp-{epoch:04d}.hdf5"

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
        num_classes=params.num_classes+1,
        dropout_rate=hyper_params.dropout_rate,
        final_layer_activation=keras.activations.softmax,
        optimizer=hyper_params.optimizer
    )

    model.summary()

    tensorboard = keras.callbacks.TensorBoard(
        log_dir=Dirs.LOG_DIR.format(sess=training_session)
    )

    # Load Checkpoints
    cp_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_path,
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

def main():
    params = Params(
        model=Consts.INCEPTIONV3,
        num_classes=10,
        examples_per_class=5,
        crop_pctg=20,
        include_unknown=True
    )
    hyper_params = HyperParams(
        num_epochs=200,
        dropout_rate=0.00001,
        batch_size=32,
        final_layer_activation=keras.activations.softmax,
        optimizer=keras.optimizers.RMSprop()
    )
    train_config = Config(
        split=Consts.TRAIN,
        include_original=True,
        include_transform=True,
        include_face_patch=True
    )
    val_config = Config(
        split=Consts.VAL,
        include_original=True,
        include_transform=True,
        include_face_patch=True
    )

    train_classifier(params, hyper_params, train_config, val_config)

if __name__ == "__main__":
    main()
