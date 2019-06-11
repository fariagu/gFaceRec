from __future__ import absolute_import, division, print_function

import platform

class Flags:
    ENV_WINDOWS = platform.system() == "Windows"

class Consts:
    NUM_CLASSES = 500
    AUG_MULT = 20
    VAL_PERCENTAGE = 10
    DROPOUT_RATE = 0.5
    BASE_LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    N_WORKERS = 1 if Flags.ENV_WINDOWS else 4
    INCEPTIONV3 = "InceptionV3"
    VGG16 = "VGG16"
    RESNET50 = "Resnet50"
    SENET50 = "SeNet50"
    MODELS = [INCEPTIONV3, VGG16, RESNET50, SENET50]
    # CROP_PCTGS = [-1, 0, 5, 10, 15, 20, 25, 30]     # -1 means no_crop
    CROP_PCTGS = [-1, 30]     # -1 means no_crop
    TRAIN = "train"
    VAL = "val"
    SPLITS = [TRAIN, VAL]
    ORIGINAL = "original"
    TRANSFORM = "transform"
    FACE_PATCH = "face_patch"
    VERSIONS = [ORIGINAL, TRANSFORM, FACE_PATCH]

    @staticmethod
    def get_image_size(model):
        if model not in Consts.MODELS:
            print("WRONG ARGS: Invalid model.")
            return -1

        if model == Consts.INCEPTIONV3:
            return (160, 160)
        else:
            return (224, 224)

class Dirs:
    ROOT_DIR = "C:/" if Flags.ENV_WINDOWS else "/home/gustavoduartefaria/"
    DATASET_DIR = "{}datasets/CelebA/".format(ROOT_DIR)
    ORIGINAL_DIR = "{}img_align_celeba/".format(DATASET_DIR)
    STRUCTURED_DIR = "{}structured_dir/".format(DATASET_DIR)
    DATASET_CACHE_DIR = "{}dataset_cache/CelebA/".format(ROOT_DIR)
    VECTORS_DIR = "{}vectors/".format(DATASET_CACHE_DIR)

    LOG_BASE_DIR = "{}logs/".format(ROOT_DIR)
    TRAINING_SESSION_PATH = "{}training_session.txt".format(LOG_BASE_DIR)
    LOG_DIR = LOG_BASE_DIR + "training_{sess:04d}/"
    PARAMS_PATH = LOG_BASE_DIR + "training_{sess:04d}/params.txt"
    TRAINING_BASE_DIR = "{}training/".format(ROOT_DIR)
    CHECKPOINT_BASE_DIR = TRAINING_BASE_DIR + "training_{sess:04d}/"
    CHECKPOINT_PATH = CHECKPOINT_BASE_DIR + "cp-{epoch:04d}.hdf5"

    @staticmethod
    def get_model_cache_dir(model):
        if model not in Consts.MODELS:
            print("WRONG ARGS: Invalid model.")
            return -1

        return "{}{}/".format(Dirs.VECTORS_DIR, model)

    @staticmethod
    def get_crop_dir(crop_pctg, cache=True, model=None):
        if cache and model is None:
            print("WRONG ARGS: For cache to be True a model must be specified.")
            return -1

        if crop_pctg not in Consts.CROP_PCTGS:
            print("WRONG ARGS: Invalid crop_pctg.")
            return -1

        if model is None:
            base_dir = Dirs.STRUCTURED_DIR
        else:
            base_dir = Dirs.get_model_cache_dir(model=model)

        crop_str = "crop_{pctg:02d}".format(pctg=crop_pctg) if crop_pctg >= 0 else "no_crop"

        return "{}{}/".format(base_dir, crop_str)

    @staticmethod
    def get_crop_dirs(cache=True, model=None):
        crop_dirs = []
        for pctg in Consts.CROP_PCTGS:
            crop_dirs.append(Dirs.get_crop_dir(pctg, cache, model))

        return crop_dirs

    @staticmethod
    def get_vector_model_crop_split_version_dir(model, crop_pctg, split, version):
        if model not in Consts.MODELS:
            print("WRONG ARGS: Invalid model")
            return -1

        if crop_pctg not in Consts.CROP_PCTGS:
            print("WRONG ARGS: invalid crop_pctg")
            return -1

        if split not in Consts.SPLITS:
            print("WRONG ARGS: invalid split")
            return -1

        if version not in Consts.VERSIONS:
            print("WRONG ARGS: invalid version")
            return -1

        crop_dir = Dirs.get_crop_dir(crop_pctg=30, model=model) # reformulei e entao a unica crop_pctg e 30

        return "{}{}/{}/".format(crop_dir, split, version)

    @staticmethod
    def get_image_crop_split_version_dir(crop_pctg, split, version):

        if crop_pctg not in Consts.CROP_PCTGS:
            print("WRONG ARGS: invalid crop_pctg")
            return -1

        if split not in Consts.SPLITS:
            print("WRONG ARGS: invalid split")
            return -1

        if version not in Consts.VERSIONS:
            print("WRONG ARGS: invalid version")
            return -1

        crop_dir = Dirs.get_crop_dir(crop_pctg=crop_pctg, cache=False)

        return "{}{}/{}/".format(crop_dir, split, version)

    @staticmethod
    def get_log_dir(training_session):
        return "{}logs/training_{sess:04d}/".format(Dirs.ROOT_DIR, sess=training_session)

def main():
    print("##################################################")
    crop_dirs = Dirs.get_crop_dirs(model=Consts.VGG16)

    for crop_dir in crop_dirs:
        print(crop_dir)

    print("##################################################")

if __name__ == "__main__":
    main()

    # x = Dirs.CHECKPOINT_PATH.format(sess=10, epoch=20)
    # z = 0
