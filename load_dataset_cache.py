from __future__ import absolute_import, division, print_function

import os

from new_utils import Dirs, Consts

def generate_cache_tree(root_dir, num_classes=Consts.NUM_CLASSES):
    
    
    if not os.path.exists(root_dir):
        print("USER ERROR: invalid root_dir value.")
        return
    
    if not os.path.exists(Dirs.get_vectors_dir()):
        os.mkdir(Dirs.get_vectors_dir())

    for model in Consts.MODELS:
        model_path = Dirs.getModelCacheDir(model)
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        crop_dirs = Dirs.getCropDirs(model=model)
        for crop_dir in crop_dirs:
            if not os.path.exists(crop_dir):
                os.mkdir(crop_dir)

            for split in Consts.SPLITS:
                split_dir = "{}{}/".format(crop_dir, split)
                if not os.path.exists(split_dir):
                    os.mkdir(split_dir)
                
                for version in Consts.VERSIONS:
                    version_dir = "{}{}/".format(split_dir, version)
                    if not os.path.exists(version_dir):
                        os.mkdir(version_dir)

def main():
    # generate_cache_tree(Dirs.DATASET_CACHE_DIR)
    # final_dir = Dirs.getVectorModelCropSplitVersionDir(
    #     model=Consts.VGG16,
    #     crop_pctg=0,
    #     split=Consts.TRAIN,
    #     version=Consts.ORIGINAL
    # )
    final_dir = Dirs.getImageCropSplitVersionDir(
        crop_pctg=0,
        split=Consts.TRAIN,
        version=Consts.ORIGINAL
    )

    print(final_dir)

if __name__ == "__main__":
    main()