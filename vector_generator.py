from __future__ import absolute_import, division, print_function

import numpy as np
import pickle

import keras

class VectorGenerator(keras.utils.Sequence):

    def __init__(self, vector_filenames, labels, batch_size):
        self.vectors, self.labels = vector_filenames, labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.labels) / float(self.batch_size)))

    def __getitem__(self, idx):
        vector_batch = self.vectors[idx * self.batch_size:(idx + 1) * self.batch_size]
        label_batch = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        vector_array = []
        for vector in vector_batch:
            with open(vector, "rb") as v:
                vector_array.append(pickle.load(v))

        return np.array(vector_array), np.array(label_batch)
