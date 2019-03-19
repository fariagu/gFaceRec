from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf

checkpoint_dir = os.path.dirname(os.path.realpath(__file__))
checkpoint_dir = checkpoint_dir.replace('\\', '/')

checkpoint_path = checkpoint_dir + "/training_3/cp-{epoch:03d}.hdf5"
model_path = checkpoint_dir + "/tflite_models/converted_model-{epoch:03d}.tflite"
# print(model_path.format(epoch=50))

converter = tf.lite.TFLiteConverter.from_keras_model_file(checkpoint_path.format(epoch=50))
tflite_model = converter.convert()
open(model_path.format(epoch=50), "wb").write(tflite_model)