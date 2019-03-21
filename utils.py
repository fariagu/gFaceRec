from __future__ import absolute_import, division, print_function

import os

env_windows = True

training_session    = 7
num_classes         = 100
batch_size          = 8
num_epochs          = 100
cp_period           = 5     # save model every <cp_period> epochs

# root directory full path (os independent i hope)
base_dir = os.path.dirname(os.path.realpath(__file__)).replace('\\', '/')

# logs and model checkpoints
log_dir = base_dir + "/logs/training_" + str(training_session) + "/"
training_dir = base_dir + "training/training_" + str(training_session) + "/"
checkpoint_path = training_dir + "cp-{epoch:04d}.hdf5"

### data directories

## OS dependent
if (env_windows):
    # windows (laptop glintt)
    raw_dir = "C:/datasets/CelebA/"
    cache_dir = "C:/dataset_cache/CelebA/"
else:
    # # rhel7 (grid.fe.up.pt)
    # raw_dir = "/homes/up201304501/datasets/CelebA/"
    # cache_dir = "/homes/up201304501/dataset_cache/CelebA/"

    # gcloud
    raw_dir = "/home/gustavoduartefaria/datasets/CelebA/"
    cache_dir = "/home/gustavoduartefaria/dataset_cache/CelebA"

## OS independent
images_dir = raw_dir + "img_align_celeba/"
labels_path = raw_dir + "identity_CelebA.txt"
partition_path = raw_dir + "list_eval_partition.txt"
cache_partition_path = cache_dir + "train_val_test.pkl"

# initialize necessary directories
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if not os.path.exists(training_dir):
    os.makedirs(training_dir)

print(base_dir)