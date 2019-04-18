import os

from multiprocessing import Pool

import detect_align
import utils
from load_celeba import filenames_and_labels

if not os.path.exists(utils.crop_dir):
    os.makedirs(utils.crop_dir)

p = Pool(utils.n_workers)

identity_dict = filenames_and_labels()
images = []
for img in os.listdir(utils.images_dir):
    if identity_dict[img] < utils.num_classes:
        images.append(img)

p.map(detect_align.main, images)
# p.map(detect_align.main, os.listdir(utils.images_dir))
# p.map(detect_align.main, os.listdir("C:/datasets/CelebA/tmp/"))