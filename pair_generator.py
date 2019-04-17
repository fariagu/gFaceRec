from __future__ import absolute_import, division, print_function

import numpy as np
import pickle

import keras

import utils

class PairGenerator(keras.utils.Sequence):

    def __init__(self, vector_filenames, labels, batch_size):
        self.vectors, self.labels = vector_filenames, labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.labels) / float(self.batch_size)))

    def __getitem__(self, idx):
        vector_batch = self.vectors[idx * self.batch_size:(idx + 1) * self.batch_size]
        label_batch = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        vector_array = []
        # two halves make a whole... get it?
        for two_halves in vector_batch:
            with open(two_halves[0], "rb") as v1:
                vector1 = pickle.load(v1)
            with open(two_halves[1], "rb") as v2:
                vector2 = pickle.load(v2)
            
            whole = np.concatenate((vector1, vector2), axis=None)

            vector_array.append(whole)
        
        return np.array(vector_array), np.array(label_batch)