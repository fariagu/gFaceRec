from __future__ import absolute_import, division, print_function

import numpy as np
import keras
from keras import layers
from keras.models import load_model

from keras.engine import  Model
from keras.layers import Flatten, Dense, Input
from keras_vggface.vggface import VGGFace

# import tensorflow_hub as hub
from new_utils import Consts

def load_face_detector():
    model = load_model("detector/yolov2_tiny-face.h5")
    model.trainable = False

    return model

def load_vgg_face_fv():
    vgg_model = VGGFace(model='vgg16', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    vgg_model.trainable = False

    return vgg_model

def load_facenet_fv():
    model = load_model("facenet/model.h5")
    model.load_weights("facenet/weights.h5")
    model.trainable = False

    return model

def load_local_fv(model):
    if model == Consts.INCEPTIONV3:
        return load_facenet_fv()
    elif model == Consts.VGG16:
        return load_vgg_face_fv()
    else:
        print("Not implemented yet")
        return None
        # TODO: resnet e senet

# def load_facenet_model():
#     base_model = load_model("facenet/model.h5")
#     base_model.load_weights("facenet/weights.h5")
#     base_model.trainable = False

#     dropout_layer = keras.layers.Dropout(utils.dropout_rate)

#     prediction_layer = keras.layers.Dense(
#         units=utils.num_classes,
#         activation=keras.activations.softmax
#     )

#     model = keras.Sequential([
#         base_model,
#         dropout_layer,
#         prediction_layer,
#     ])

#     model.compile(
#         optimizer=keras.optimizers.Adam(
#             lr=utils.base_learning_rate
#         ),
#         loss=keras.losses.sparse_categorical_crossentropy,
#         metrics=['accuracy']
#     )

#     return model

# def load_vgg16(num_classes=Consts.NUM_CLASSES, dropout_rate=Consts.DROPOUT_RATE, base_learning_rate=Consts.BASE_LEARNING_RATE):
#     vgg_model = VGGFace(model="vgg16", include_top=False, input_shape=(224, 224, 3), pooling='avg')
#     vgg_model.trainable = False

#     dropout_layer = keras.layers.Dropout(dropout_rate)

#     prediction_layer = keras.layers.Dense(
#         units=num_classes,
#         activation=keras.activations.softmax
#     )

#     model = keras.Sequential([
#         vgg_model,
#         dropout_layer,
#         prediction_layer,
#     ])

#     model.compile(
#         optimizer=keras.optimizers.Adam(
#             lr=base_learning_rate
#         ),
#         loss=keras.losses.sparse_categorical_crossentropy,
#         metrics=['accuracy']
#     )

#     return model

# def load_local_model(model):
#     if model == utils.FACENET:
#         return load_facenet_model()
#     elif model == utils.VGGFACE:
#         return load_vgg_face()
#     else:
#         return load_mobilenet_model()
#         # se quiser retomar checkpoint:
#         # model.load_weights("./mobilenet/model.hdf5")