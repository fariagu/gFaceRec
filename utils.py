from __future__ import absolute_import, division, print_function

import os
import platform

import keras

def read_training_session():
    r = open("training_session.txt", "r")
    training_session = int(r.read())
    print("Training Session " + str(training_session))
    r.close()
    w = open("training_session.txt", "w")
    w.write(str(training_session+1))
    w.close()

    return training_session

def save_session_params():
    f = open("logs/training_{sess:04d}/params.txt".format(sess=training_session), "w")
    f.write("Session: " + str(training_session) + "\n")
    f.write("Base learning rate: " + str(base_learning_rate) + "\n")
    f.write("Dropout rate: " + str(dropout_rate) + "\n")
    f.write("Number of classes: " + str(num_classes) + "\n")
    f.write("Batch Size: " + str(batch_size) + "\n")
    f.write("Epochs: " + str(num_epochs) + "\n")
    f.write("Model: " + model_in_use + "\n")
    f.write("Optimizer: " + optimizer + "\n")
    f.write("Final Layer Activation: " + activation + "\n")
    f.write("Loss: " + loss + "\n")
    f.write("Env: " + platform.system() + "\n")
    f.write("AUG: TRUE\n") if AUGMENTATION == True else f.write("AUG: FALSE\n")
    f.write("CROPPED: TRUE\n") if CROPPED == True else f.write("CROPPED: FALSE\n")

env_windows = True if platform.system() == "Windows" else False

training_session    = read_training_session()
base_learning_rate  = 0.025
dropout_rate        = 0.5
num_classes         = 100       # full dataset: 10177
batch_size          = 1024
num_epochs          = 200
cp_period           = num_epochs / 10        # save model every <cp_period> epochs

optimizers = {
    "rmsprop": keras.optimizers.RMSprop(lr=base_learning_rate),
    "adam": keras.optimizers.Adam(lr=base_learning_rate),
    "adagrad": keras.optimizers.Adagrad(lr=base_learning_rate),
    "adadelta": keras.optimizers.Adadelta(lr=base_learning_rate),
}

activations = {
    "sigmoid": keras.activations.sigmoid,
    "softmax": keras.activations.softmax,
    "relu": keras.activations.relu,
}

losses = {
    "binary_crossentropy": keras.losses.binary_crossentropy,
    "categorical_crossentropy": keras.losses.categorical_crossentropy,
    "sparse_crossentropy": keras.losses.sparse_categorical_crossentropy,
}

# choose optimizer, activation and loss here
optimizer = "rmsprop"
activation = "sigmoid"
loss = "binary_crossentropy"

FACENET = "Facenet"
MOBILENET = "Mobilenet"
VGGFACE = "VggFace"

AUGMENTATION = False
CROPPED = False

# [0, 100] for when cropping the dataset
margin_percentage = 20

# FACENET || MOBILENET || VGGFACE
model_in_use = VGGFACE
image_width = 160

if model_in_use == FACENET:
    image_width = 160
elif model_in_use == VGGFACE:
    image_width = 224
else:
    image_width = 224

mobilenet_feature_vector_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/2"

# root directory full path (os independent)
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
    # gcloud
    raw_dir = "/home/gustavoduartefaria/datasets/CelebA/"
    cache_dir = "/home/gustavoduartefaria/dataset_cache/CelebA/"

    multiprocessing = True
    n_workers = 22

## OS independent
images_dir = raw_dir + "img_align_celeba/"
crop_dir = raw_dir + "img_crop/" if margin_percentage == 0 else raw_dir + "img_crop_" + str(margin_percentage) + "/"
labels_path = raw_dir + "identity_CelebA.txt"
partition_path = raw_dir + "list_eval_partition.txt"
cache_partition_path = cache_dir + "train_val_test.pkl"

if CROPPED:
    images_dir = crop_dir

# augmentation dirs
split_dir = raw_dir + "split_" + str(num_classes) + "/"
train_dir = split_dir + "train/"
val_dir = split_dir + "val/"
test_dir = split_dir + "test/"
train_dir_aug = split_dir + "train_aug/"

# extracted info dirs
identity_cache_dir = cache_dir + "identity_dict.pkl"
vector_dir = cache_dir + model_in_use + "_vectors"

vector_dir = vector_dir + "_crop_" + margin_percentage + "/" if CROPPED else vector_dir + "/"

# initialize necessary directories
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

if not os.path.exists(training_dir):
    os.makedirs(training_dir)

save_session_params()

print(base_dir)