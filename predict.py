from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf

import numpy as np
import keras
from keras.models import load_model

import utils
from load_celeba import load_test_data
from generator import Generator
from finetune import load_local_model

test_images, test_labels = load_test_data()
test_batch_generator = Generator(test_images, test_labels, utils.batch_size)

model = load_local_model()
if utils.model_in_use == utils.FACENET:
    model.load_weights("./facenet/fn-70-no_aug.hdf5")
    pass
else:
    model.load_weights("./mobilenet/model.hdf5")

result = model.predict_generator(test_batch_generator, verbose=1)

correct_guesses = 0
for i, prediction in enumerate(result):
    guess = np.argmax(prediction)
    answer = test_labels[i]

    if guess == answer:
        correct_guesses += 1
    else:
        print("wrong guess")
        print("G: " + str(prediction[guess]))
        print("A: " + str(prediction[answer]))

test_score = correct_guesses * 100 / len(test_labels)

print(test_score)


# len(test_labels) ---- 100%
# correct_guesses ------ x%