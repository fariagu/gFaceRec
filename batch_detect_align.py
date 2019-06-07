import os

from multiprocessing import Pool

import detect_align
import utils
from load_celeba import filenames_and_labels, load_vectors_into_disk

if not os.path.exists(utils.crop_dir):
    os.makedirs(utils.crop_dir)

for label in os.listdir("structured dir"):
    if img in identity_dict.keys():
        if identity_dict[img] < utils.num_classes:
            images.append(img)

p = Pool(utils.n_workers)
p.map(detect_align.main, images)
p.close()

print("DONE")