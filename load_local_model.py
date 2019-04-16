from __future__ import absolute_import, division, print_function

import numpy as np
import keras
from keras import layers
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from keras.engine import  Model
from keras.layers import Flatten, Dense, Input
from keras_vggface.vggface import VGGFace

import tensorflow as tf
import tensorflow_hub as hub

import utils

def load_vgg_face():
    vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3), pooling='avg')
    vgg_model.trainable = False
    # vgg_model.summary()
    # last_layer = vgg_model.get_layer('avg_pool').output
    # x = Flatten(name='flatten')#(last_layer)
    # out = Dense(utils.num_classes, activation='softmax', name='classifier')(x)
    # return Model(vgg_model.input, out)

    dropout_layer = keras.layers.Dropout(utils.dropout_rate)

    prediction_layer = keras.layers.Dense(
        units=utils.num_classes,
        activation=keras.activations.softmax
    )

    model = keras.Sequential([
        vgg_model,
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

def load_vgg_face_fv():
    vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3), pooling='avg')
    vgg_model.trainable = False

    return vgg_model

def load_facenet_fv():
    model = load_model("facenet/model.h5")
    model.load_weights("facenet/weights.h5")
    # true maybe se a euclidean distance se portar bem? treino limitadamente as camadas mais superficiais (tipo uma noite inteira a treinar e vejo se valeu a pena) (mas como é que treino um feature vector tho=? fds)
    model.trainable = False

    return model

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
    elif utils.model_in_use == utils.VGGFACE:
        return load_vgg_face()
    else:
        return load_mobilenet_model()
        # se quiser retomar checkpoint:
        # model.load_weights("./mobilenet/model.hdf5")