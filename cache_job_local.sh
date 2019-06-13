#!/bin/bash

###################

python -W ignore load_dataset_cache.py VGG16 25 train original

python -W ignore load_dataset_cache.py VGG16 25 train transform

python -W ignore load_dataset_cache.py VGG16 25 train face_patch

python -W ignore load_dataset_cache.py VGG16 25 val original

python -W ignore load_dataset_cache.py VGG16 25 val transform

python -W ignore load_dataset_cache.py VGG16 25 val face_patch

###################

python -W ignore load_dataset_cache.py VGG16 10 train original

python -W ignore load_dataset_cache.py VGG16 10 train transform

python -W ignore load_dataset_cache.py VGG16 10 train face_patch

python -W ignore load_dataset_cache.py VGG16 10 val original

python -W ignore load_dataset_cache.py VGG16 10 val transform

python -W ignore load_dataset_cache.py VGG16 10 val face_patch

###################

python -W ignore load_dataset_cache.py VGG16 5 train original

python -W ignore load_dataset_cache.py VGG16 5 train transform

python -W ignore load_dataset_cache.py VGG16 5 train face_patch

python -W ignore load_dataset_cache.py VGG16 5 val original

python -W ignore load_dataset_cache.py VGG16 5 val transform

python -W ignore load_dataset_cache.py VGG16 5 val face_patch

###################

# python -W ignore load_dataset_cache.py VGG16 0 train original #

# python -W ignore load_dataset_cache.py VGG16 0 train transform # confirmar

python -W ignore load_dataset_cache.py VGG16 0 train face_patch

python -W ignore load_dataset_cache.py VGG16 0 val original

python -W ignore load_dataset_cache.py VGG16 0 val transform

python -W ignore load_dataset_cache.py VGG16 0 val face_patch

###################

python -W ignore load_dataset_cache.py VGG16 30 train original

python -W ignore load_dataset_cache.py VGG16 30 train transform

python -W ignore load_dataset_cache.py VGG16 30 train face_patch

python -W ignore load_dataset_cache.py VGG16 30 val original

python -W ignore load_dataset_cache.py VGG16 30 val transform

python -W ignore load_dataset_cache.py VGG16 30 val face_patch

###################

python -W ignore load_dataset_cache.py InceptionV3 15 train transform
