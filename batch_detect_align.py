import os

from multiprocessing import Pool

import detect_align
import utils
from load_celeba import filenames_and_labels, load_vectors_into_disk

if not os.path.exists(utils.crop_dir):
    os.makedirs(utils.crop_dir)

# print(utils.og_images_dir)

identity_dict = filenames_and_labels()
images = []
for img in os.listdir(utils.og_images_dir):
# for img in os.listdir("C:/datasets/CelebA/img_crop/"):
    if img in identity_dict.keys():
        if identity_dict[img] < utils.num_classes:
            images.append(img)

p = Pool(utils.n_workers)
p.map(detect_align.main, images)
# p.map(detect_align.main, os.listdir(utils.og_images_dir))
# p.map(detect_align.main, os.listdir("C:/datasets/CelebA/tmp/"))
p.close()

load_vectors_into_disk()

print("DONE")