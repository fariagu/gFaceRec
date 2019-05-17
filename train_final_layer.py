from __future__ import absolute_import, division, print_function

import keras

from load_dataset_cache import vectors_and_labels
from vector_generator import VectorGenerator
from load_local_model import classifier

from new_utils import Consts, Dirs

def train_classifier(model, crop_pctg, train_split, val_split, version):
    train_paths, train_labels = vectors_and_labels(model, crop_pctg, train_split, version)
    val_paths, val_labels = vectors_and_labels(model, crop_pctg, val_split, version)

    train_generator = VectorGenerator(
        train_paths,
        train_labels,
        Consts.BATCH_SIZE
    )

    val_generator = VectorGenerator(
        val_paths,
        val_labels,
        Consts.BATCH_SIZE
    )

    model = classifier(
        fv_len=128,
        num_classes=10,
        dropout_rate=0.5,
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
        period=utils.cp_period
    )

    model.save_weights(Dirs.CHECKPOINT_PATH.format(sess=training_session, epoch=0))

    model.fit_generator(
        generator=train_generator,
        epochs=utils.num_epochs,
        callbacks=[cp_callback, tensorboard],
        verbose=1,
        validation_data=val_generator,
        use_multiprocessing=utils.multiprocessing,
        workers=utils.n_workers,
        shuffle=False, # I do that already
    )