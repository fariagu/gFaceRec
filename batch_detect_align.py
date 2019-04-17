import os

from multiprocessing import Pool

import detect_align
import utils

if not os.path.exists(utils.crop_dir):
    os.makedirs(utils.crop_dir)

p = Pool(utils.n_workers)
p.map(detect_align.main, os.listdir(utils.images_dir))
# p.map(detect_align.main, os.listdir("C:/datasets/CelebA/tmp/"))