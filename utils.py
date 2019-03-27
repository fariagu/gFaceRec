from __future__ import absolute_import, division, print_function

import os

def read_training_session():
    r = open("training_session.txt", "r")
    training_session = int(r.read())
    print("Training Session " + training_session)
    r.close()
    w = open("training_session.txt", "w")
    w.write(str(training_session+1))
    w.close()

    return training_session

def save_session_params():
    f = open("logs/training_{sess:04d}.txt".format(sess=training_session))

env_windows = False

training_session    = read_training_session()
base_learning_rate  = 1.0
num_classes         = 100       # full dataset: 10177
batch_size          = 32
num_epochs          = 100
cp_period           = 10        # save model every <cp_period> epochs

# facenet
# image_width = 160

# mobilenet
image_width = 224

# root directory full path (os independent i hope)
base_dir = os.path.dirname(os.path.realpath(__file__)).replace('\\', '/')

# logs and model checkpoints
log_dir = base_dir + "/logs/training_" + "{sess:04d}/".format(sess=training_session)
training_dir = base_dir + "/training/training_" + "{sess:04d}/".format(sess=training_session)
checkpoint_path = training_dir + "cp-{epoch:04d}.hdf5"

### data directories

## OS dependent
if (env_windows):
    # windows (laptop glintt)
    raw_dir = "C:/datasets/CelebA/"
    cache_dir = "C:/dataset_cache/CelebA/"

    multiprocessing = False
    n_workers = 1
else:
    # # rhel7 (grid.fe.up.pt)
    # raw_dir = "/homes/up201304501/datasets/CelebA/"
    # cache_dir = "/homes/up201304501/dataset_cache/CelebA/"

    # gcloud
    raw_dir = "/home/gustavoduartefaria/datasets/CelebA/"
    cache_dir = "/home/gustavoduartefaria/dataset_cache/CelebA/"

    multiprocessing = True
    n_workers = 22

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