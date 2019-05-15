from __future__ import absolute_import, division, print_function

import platform
import os

class Flags:
    ENV_WINDOWS = platform.system() == "Windows"

class Consts:
    NUM_CLASSES = 500
    AUG_MULT = 10
    VAL_PERCENTAGE = 10
    CROP_PCTGS = [-1, 0, 5, 10, 15, 20, 25, 30]     # -1 means no_crop
    INCEPTIONV3 = "InceptionV3"
    VGG16 = "VGG16"
    RESNET50 = "Resnet50"
    SENET50 = "SeNet50"
    MODELS = [INCEPTIONV3, VGG16, RESNET50, SENET50]

class Dirs:
    ROOT_DIR = "C:/" if Flags.ENV_WINDOWS else "/home/gustavoduartefaria/"
    DATASET_DIR = "{}datasets/CelebA/".format(ROOT_DIR)
    ORIGINAL_DIR = "{}img_celeba/".format(DATASET_DIR)
    DATASET_CACHE_DIR = "{}dataset_cache/CelebA/".format(ROOT_DIR)

    @staticmethod
    def get_raw_filter_dir(num_classes=Consts.NUM_CLASSES):
        return "{}filter_{}_classes/".format(Dirs.DATASET_DIR, num_classes)

    @staticmethod
    def get_vectors_dir(num_classes=Consts.NUM_CLASSES):
        return "{}vectors_{}/".format(Dirs.DATASET_CACHE_DIR, num_classes)

    @staticmethod
    def getModelCacheDir(model, num_classes=Consts.NUM_CLASSES):
        if model not in Consts.MODELS:
            print("WRONG ARGS: Invalid model.")
            return

        return "{}{}/".format(Dirs.get_vectors_dir(num_classes), model)

    @staticmethod
    def getCropDir(crop_pctg, cache=True, model=None, num_classes=Consts.NUM_CLASSES):
        if cache and model == None:
            print("WRONG ARGS: For cache to be True a model must be specified.")
            return
        
        if crop_pctg not in Consts.CROP_PCTGS:
            print("WRONG ARGS: Invalid crop_pctg.")
            return

        base_dir = Dirs.getModelCacheDir(model, num_classes) if model != None else Dirs.get_raw_filter_dir(num_classes)

        crop_str = "crop_{pctg:02d}".format(pctg=crop_pctg) if crop_pctg >= 0 else "no_crop"

        return "{}{}/".format(base_dir, crop_str)
    
    @staticmethod
    def getCropDirs(cache=True, model=None, num_classes=Consts.NUM_CLASSES):
        crop_dirs = []
        for pctg in Consts.CROP_PCTGS:
            crop_dirs.append(Dirs.getCropDir(pctg, cache, model, num_classes))
        
        return crop_dirs

# for testing
def main():
    print("##################################################")
    # print(Dirs.getCropDir(crop_pctg=0, model=Consts.INCEPTIONV3))
    # print(dirs.getCropDir(crop_pctg=0, cache=False))
    # crop_dirs = Dirs.getCropDirs(cache=False)
    crop_dirs = Dirs.getCropDirs(model=Consts.INCEPTIONV3)

    for crop_dir in crop_dirs:
        print(crop_dir)
    
    print("##################################################")

if __name__ == "__main__":
    main()
