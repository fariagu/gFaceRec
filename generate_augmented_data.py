from __future__ import absolute_import, division, print_function

from finetune import finetune
from load_celeba import load_vectors_into_disk
from simple_svm import svm

aug_mult = 10

# finetune(aug_mult)
# load_vectors_into_disk()
svm()