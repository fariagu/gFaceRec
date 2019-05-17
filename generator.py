from __future__ import absolute_import, division, print_function

from skimage.transform import resize
import numpy as np

from new_utils import Consts

import keras

class Generator(keras.utils.Sequence):

    def __init__(self, image_filenames, labels, batch_size, model):
        self.images, self.labels = image_filenames, labels
        self.batch_size = batch_size
        self.model = model

    def __len__(self):
        return int(np.ceil(len(self.labels) / float(self.batch_size)))

    def __getitem__(self, idx):
        image_batch = self.images[idx * self.batch_size:(idx + 1) * self.batch_size]
        label_batch = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        image_array = []
        for file_name in image_batch:
            img = resize(
                imread(file_name),
                (Consts.get_image_size(self.model))
            )

            if self.model == Consts.VGG16:
                np.transpose(img, (2, 0, 1))

            image_array.append(img)
            
        return np.array(image_array), np.array(label_batch)