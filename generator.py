from __future__ import absolute_import, division, print_function

from skimage.transform import resize
from skimage.io import imread
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from new_utils import Consts

import keras

class Generator(keras.utils.Sequence):

    def __init__(self, image_filenames, labels, batch_size, model, crop_pctg):
        self.images, self.labels = image_filenames, labels
        self.batch_size = batch_size
        self.model = model
        self.crop_pctg = crop_pctg

    def __len__(self):
        return int(np.ceil(len(self.labels) / float(self.batch_size)))

    def __getitem__(self, idx):
        image_batch = self.images[idx * self.batch_size:(idx + 1) * self.batch_size]
        label_batch = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        image_array = []
        (height, width) = Consts.get_image_size(self.model)
        for file_name in image_batch:
            img = resize(
                imread(file_name),
                (height, width)
            )

            if self.model == Consts.VGG16:
                np.transpose(img, (2, 0, 1))

            crop_len = self.get_crop_len(width)

            if crop_len > 0:
                img = img[crop_len:-crop_len, crop_len:-crop_len]

            image_array.append(img)

        return np.array(image_array), np.array(label_batch)

    def get_crop_len(self, length):
        pctg = 30 - self.crop_pctg

        return pctg * length / 100

# def get_crop_len(length, crop_pctg):
#     pctg = 30 - crop_pctg

#     return int(pctg * length / 100)

# if __name__ == "__main__":
#     tmp = resize(
#         imread("C:/datasets/CelebA/crop_30/train/original/003636.jpg"),
#         (224, 224)
#     )

#     (heighht, whidth) = (224, 224)
#     crop_lhen = get_crop_len(whidth, 0)

#     if crop_lhen > 0:
#         tmp = tmp[crop_lhen:-crop_lhen, crop_lhen:-crop_lhen]

#     tmpplot = plt.imshow(tmp)
#     plt.show()

#     x = 0
