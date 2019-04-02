import os

import detect_align
import utils

if not os.path.exists(utils.crop_dir):
    os.makedirs(utils.crop_dir)

for file in os.listdir(utils.images_dir):
    detect_align.main(utils.images_dir + file, file)