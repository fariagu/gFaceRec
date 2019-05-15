from __future__ import absolute_import, division, print_function

import platform
import os

class Flags:
    ENV_WINDOWS = platform.system() == "Windows"

class Consts:
    NUM_CLASSES = 500
    AUG_MULT = 10
    VAL_PERCENTAGE = 10
    INCEPTIONV3 = "InceptionV3"
    VGG16 = "VGG16"
    RESNET50 = "Resnet50"
    SENET50 = "SeNet50"
    MODELS = [INCEPTIONV3, VGG16, RESNET50, SENET50]
    CROP_PCTGS = [-1, 0, 5, 10, 15, 20, 25, 30]     # -1 means no_crop
    TRAIN = "train"
    VAL = "val"
    SPLITS = [TRAIN, VAL]
    ORIGINAL = "original"
    TRANSFORM = "transform"
    FACE_PATCH = "face_patch"
    VERSIONS = [ORIGINAL, TRANSFORM, FACE_PATCH]

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

        base_dir = Dirs.getModelCacheDir(model=model, num_classes=num_classes) if model != None else Dirs.get_raw_filter_dir(num_classes)

        crop_str = "crop_{pctg:02d}".format(pctg=crop_pctg) if crop_pctg >= 0 else "no_crop"

        return "{}{}/".format(base_dir, crop_str)
    
    @staticmethod
    def getCropDirs(cache=True, model=None, num_classes=Consts.NUM_CLASSES):
        crop_dirs = []
        for pctg in Consts.CROP_PCTGS:
            crop_dirs.append(Dirs.getCropDir(pctg, cache, model, num_classes))
        
        return crop_dirs
    
    @staticmethod
    def getVectorModelCropSplitVersionDir(model, crop_pctg, split, version, num_classes=Consts.NUM_CLASSES):
        vectors_dir = Dirs.get_vectors_dir(num_classes)

        if not os.path.exists(vectors_dir):
            print("WRONG ARGS: invalid num_classes")
            return
        
        if model not in Consts.MODELS:
            print("WRONG ARGS: Invalid model")
            return
        
        if crop_pctg not in Consts.CROP_PCTGS:
            print("WRONG ARGS: invalid crop_pctg")
            return
        
        if split not in Consts.SPLITS:
            print("WRONG ARGS: invalid split")
            return
        
        if version not in Consts.VERSIONS:
            print("WRONG ARGS: invalid version")
            return
        
        crop_dir = Dirs.getCropDir(crop_pctg=crop_pctg, model=model, num_classes=num_classes)
        
        return "{}{}/{}/".format(crop_dir, split, version)
    
    @staticmethod
    def getImageCropSplitVersionDir(crop_pctg, split, version, num_classes=Consts.NUM_CLASSES):

        if crop_pctg not in Consts.CROP_PCTGS:
            print("WRONG ARGS: invalid crop_pctg")
            return
        
        if split not in Consts.SPLITS:
            print("WRONG ARGS: invalid split")
            return
        
        if version not in Consts.VERSIONS:
            print("WRONG ARGS: invalid version")
            return
        
        crop_dir = Dirs.getCropDir(crop_pctg=crop_pctg, cache=False, num_classes=num_classes)
        
        return "{}{}/{}/".format(crop_dir, split, version)


# for testing
def main():
    print("##################################################")
    # print(Dirs.getCropDir(crop_pctg=0, model=Consts.VGG16))
    # print(dirs.getCropDir(crop_pctg=0, cache=False))
    # crop_dirs = Dirs.getCropDirs(cache=False)
    crop_dirs = Dirs.getCropDirs(model=Consts.VGG16)

    for crop_dir in crop_dirs:
        print(crop_dir)
    
    print("##################################################")

if __name__ == "__main__":
    main()
