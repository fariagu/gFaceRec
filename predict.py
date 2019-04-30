from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf

from skimage.io import imread
from skimage.transform import resize
from skimage import data
from skimage.viewer import ImageViewer
import yolo_v3_interpreter as yolo

import cv2

import numpy as np
import keras
from keras.models import load_model

import utils
from load_celeba import load_test_data
from generator import Generator
from load_local_model import load_local_model, load_facenet_fv, load_face_detector

test_images, test_labels = load_test_data()
test_batch_generator = Generator(test_images, test_labels, utils.batch_size)

# model = load_facenet_fv()
model = load_face_detector()

# # caga nisto para já. quero ver como é só com o feature vector
# 
# model = load_local_model()
# 
# if utils.model_in_use == utils.FACENET:
#     model.load_weights("./facenet/fn-70-no_aug.hdf5")
#     pass
# else:
#     model.load_weights("./mobilenet/model.hdf5")
# 
# result = model.predict_generator(test_batch_generator, verbose=1)

model.summary()

# img = resize(imread("C:/datasets/CelebA/img_align_celeba/000001.jpg"),(416, 416))
# img = cv2.resize(cv2.imread("C:/datasets/CelebA/img_align_celeba/001197.jpg"), (416, 416))
img = cv2.resize(cv2.imread("C:/Users/gustavo.faria/Desktop/training_summaries/people.jpg"), (416, 416))
nimg = np.zeros((416, 416))
nimg = cv2.normalize(img, nimg, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)

imgs = np.array([nimg])
# print(imgs.shape)

result = model.predict(imgs, verbose=1)
res = yolo.interpret_output_yolov2(result[0], 416, 416)

print(res)
# print(result)
# print(nimg)

for face in res:
    left = int(face[2] - (face[4] / 2))
    right = int(face[2] + (face[4] / 2))
    top = int(face[1] - (face[3] / 2))
    bottom = int(face[1] + (face[3] / 2))

    tl = (top, left)
    br = (bottom, right)

    #tmp
    cv2.rectangle(img, tl, br, (0, 0, 255), 3)

cv2.imshow("predict", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# correct_guesses = 0
# for i, prediction in enumerate(result):
#     if i < 1:
#         print(prediction)

#     guess = np.argmax(prediction)
#     answer = test_labels[i]

#     if guess == answer:
#         correct_guesses += 1
#     else:
#         print("wrong guess")
#         print("G: " + str(prediction[guess]))
#         print("A: " + str(prediction[answer]))

# test_score = correct_guesses * 100 / len(test_labels)

# print(test_score)


# len(test_labels) ---- 100%
# correct_guesses ------ x%