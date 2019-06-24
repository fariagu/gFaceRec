#!/bin/bash

# args: MODEL NUM_CLASSES EX_P_CLASS CROP_PCTG NTE OTE NVE OVE

# python -W ignore vector_distance.py VGG16 50 1 20 True True True True

# python -W ignore vector_distance.py VGG16 50 2 20 True True True True

# python -W ignore vector_distance.py VGG16 50 3 20 True True True True

# python -W ignore vector_distance.py VGG16 50 4 20 True True True True

# python -W ignore vector_distance.py VGG16 50 5 20 True True True True

# python -W ignore vector_distance.py VGG16 50 9999 20 True True True True

# python -W ignore vector_distance.py VGG16 100 1 20 True True True True

# python -W ignore vector_distance.py VGG16 100 2 20 True True True True

# python -W ignore vector_distance.py VGG16 100 3 20 True True True True

# python -W ignore vector_distance.py VGG16 100 4 20 True True True True

# python -W ignore vector_distance.py VGG16 100 5 20 True True True True

# python -W ignore vector_distance.py VGG16 100 9999 20 True True True True

# python -W ignore vector_distance.py VGG16 250 1 20 True True True True

# python -W ignore vector_distance.py VGG16 250 2 20 True True True True

# python -W ignore vector_distance.py VGG16 250 3 20 True True True True

# python -W ignore vector_distance.py VGG16 250 4 20 True True True True

# python -W ignore vector_distance.py VGG16 250 5 20 True True True True

# python -W ignore vector_distance.py VGG16 250 9999 20 True True True True

# python -W ignore vector_distance.py VGG16 500 1 20 True True True True

# python -W ignore vector_distance.py VGG16 500 2 20 True True True True

# python -W ignore vector_distance.py VGG16 500 3 20 True True True True

# python -W ignore vector_distance.py VGG16 500 4 20 True True True True

# python -W ignore vector_distance.py VGG16 500 5 20 True True True True
# #38^^
# python -W ignore vector_distance.py VGG16 500 9999 20 True True True True

# ##############################

# python -W ignore vector_distance.py InceptionV3 50 1 25 True True True True

# python -W ignore vector_distance.py InceptionV3 50 2 25 True True True True

# python -W ignore vector_distance.py InceptionV3 50 3 25 True True True True

# python -W ignore vector_distance.py InceptionV3 50 4 25 True True True True

# python -W ignore vector_distance.py InceptionV3 50 5 25 True True True True

# python -W ignore vector_distance.py InceptionV3 50 9999 25 True True True True

# python -W ignore vector_distance.py InceptionV3 100 1 25 True True True True

# python -W ignore vector_distance.py InceptionV3 100 2 25 True True True True

# python -W ignore vector_distance.py InceptionV3 100 3 25 True True True True

# python -W ignore vector_distance.py InceptionV3 100 4 25 True True True True

# python -W ignore vector_distance.py InceptionV3 100 5 25 True True True True

# python -W ignore vector_distance.py InceptionV3 100 9999 25 True True True True

# python -W ignore vector_distance.py InceptionV3 250 1 25 True True True True

# python -W ignore vector_distance.py InceptionV3 250 2 25 True True True True

# python -W ignore vector_distance.py InceptionV3 250 3 25 True True True True

# python -W ignore vector_distance.py InceptionV3 250 4 25 True True True True
# #54^^
# python -W ignore vector_distance.py InceptionV3 250 5 25 True True True True

# python -W ignore vector_distance.py InceptionV3 250 9999 25 True True True True

# python -W ignore vector_distance.py InceptionV3 500 1 25 True True True True

# python -W ignore vector_distance.py InceptionV3 500 2 25 True True True True

# python -W ignore vector_distance.py InceptionV3 500 3 25 True True True True

# python -W ignore vector_distance.py InceptionV3 500 4 25 True True True True

# python -W ignore vector_distance.py InceptionV3 500 5 25 True True True True

# python -W ignore vector_distance.py InceptionV3 500 9999 25 True True True True
#63^^
###############################
###############################

python -W ignore vector_distance.py VGG16 100 9999 20 True False True False

python -W ignore vector_distance.py VGG16 100 9999 20 True False False True

python -W ignore vector_distance.py VGG16 100 9999 20 False True True False

python -W ignore vector_distance.py VGG16 100 9999 20 False True False True

###############################

python -W ignore vector_distance.py InceptionV3 100 9999 25 True False True False

python -W ignore vector_distance.py InceptionV3 100 9999 25 True False False True

python -W ignore vector_distance.py InceptionV3 100 9999 25 False True True False

python -W ignore vector_distance.py InceptionV3 100 9999 25 False True False True