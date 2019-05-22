from __future__ import absolute_import, division, print_function

from new_utils import Consts

class Config:
    def __init__(self, split, include_original=False, include_transform=False, include_face_patch=False):
        self.split = split
        self.include_original = include_original
        self.include_transform = include_transform
        self.include_face_patch = include_face_patch

        self.list_versions = [
            (self.include_original, Consts.ORIGINAL),
            (self.include_transform, Consts.TRANSFORM),
            (self.include_face_patch, Consts.FACE_PATCH),
        ]

class Params:
    def __init__(self, model, num_classes, examples_per_class, crop_pctg):
        """examples_per_class represents maximum number of examples per class.
        If fewer are available, all examples are added.
        """
        self.model = model
        self.num_classes = num_classes
        self.examples_per_class = examples_per_class
        self.crop_pctg = crop_pctg

class HyperParams:
    def __init__(self, num_epochs, dropout_rate, batch_size, final_layer_activation, optimizer):
        self.num_epochs = num_epochs
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.final_layer_activation = final_layer_activation
        self.optimizer = optimizer
