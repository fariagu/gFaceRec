from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf

import cv2

import numpy as np
import keras
from keras.models import load_model

import utils
from load_local_model import load_facenet_fv, load_vgg_face_fv
from model import facenet_svm, vgg_svm


vgg_layer = load_vgg_face_fv()

prediction_layer = vgg_svm()
prediction_layer.load_weights("/home/gustavoduartefaria/gFaceRec/training/training_0481/cp-0200.hdf5")

model = keras.Sequential([
    vgg_layer,
    prediction_layer
])

model.compile(
    optimizer=keras.optimizers.Adam(
        lr=utils.base_learning_rate
    ),
    loss=keras.losses.sparse_categorical_crossentropy,
    metrics=['accuracy']
)

model.summary()

model.save('/home/gustavoduartefaria/stacked_model_sheer.hdf5')