from __future__ import absolute_import, division, print_function

import utils
from finetune import finetune
from load_celeba import load_vectors_into_disk

aug_mult = utils.AUG_MULT

finetune(aug_mult)
load_vectors_into_disk(utils.created_aug_imgs)

#stack models
#tflite convert