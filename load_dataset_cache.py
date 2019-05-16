from __future__ import absolute_import, division, print_function

import os
from random import shuffle
import pickle


from load_local_model import load_local_fv
from generator import Generator
from new_utils import Dirs, Consts, Flags

def generate_cache_tree(root_dir):
    
    
    if not os.path.exists(root_dir):
        print("USER ERROR: invalid root_dir value.")
        return
    
    if not os.path.exists(Dirs.VECTORS_DIR):
        os.mkdir(Dirs.VECTORS_DIR)

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

def filenames_and_labels(model, crop_pctg, split, version):
    input_dir = Dirs.getImageCropSplitVersionDir(
        crop_pctg=crop_pctg,
        split=split,
        version=version
    )

    filepaths_and_labels = []
    for identity in os.listdir(input_dir):
        iden_dir = "{}{}/".format(input_dir, identity)

        for file in os.listdir(iden_dir):
            file_path = "{}{}".format(iden_dir, file)
            filepaths_and_labels.append((file_path, identity))
        
    # shuffle not necessary, this is only fed into a model for prediction, not training
    # shuffle(filepaths_and_labels)

    return zip(*filepaths_and_labels)

def generateVectors(model, crop_pctg, split, version):
    filenames, labels = filenames_and_labels(
        model=model,
        crop_pctg=crop_pctg,
        split=split,
        version=version
    )

    output_dir = Dirs.getVectorModelCropSplitVersionDir(
        model=model,
        crop_pctg=crop_pctg,
        split=split,
        version=version
    )

    generator = Generator(filenames, labels, Consts.BATCH_SIZE, model)

    keras_model = load_local_fv(model)
    # keras_model.summary()

    predictions = keras_model.predict_generator(
        generator=generator,
        verbose=1,
        use_multiprocessing=not Flags.ENV_WINDOWS,
        workers=Consts.N_WORKERS,
    )

    for fv, file_path, label in zip(predictions, filenames, labels):
        iden_dir = "{}{}/".format(output_dir, label)
        if not os.path.exists(iden_dir):
            os.mkdir(iden_dir)

        file_name = file_path.split("/")[-1].split(".")[0]
        save_path = "{}{}.pkl".format(iden_dir, file_name)
        
        with open(save_path, "wb") as v:
            pickle.dump(fv, v)


def main():
    # generate_cache_tree(Dirs.DATASET_CACHE_DIR)
    
    # final_dir = Dirs.getVectorModelCropSplitVersionDir(
    #     model=Consts.VGG16,
    #     crop_pctg=0,
    #     split=Consts.TRAIN,
    #     version=Consts.ORIGINAL
    # )
    
    # final_dir = Dirs.getImageCropSplitVersionDir(
    #     crop_pctg=0,
    #     split=Consts.TRAIN,
    #     version=Consts.ORIGINAL
    # )

    # print(final_dir)

    for model in Consts.MODELS[:2]: # porque ainda só tenho dois modelos
        for crop_pctg in Consts.CROP_PCTGS: #[1:] porque o no_crop merdou
            for split in Consts.SPLITS:
                # for version in Consts.VERSIONS:
                    # generateVectors(model, crop_pctg, split, version)
                version = Consts.VERSIONS[0] # enquanto so estou no windows
                print("GENERATING VECTORS FOR {} crop_{p:02d} {} {}".format(model, p=crop_pctg, split, version))
                generateVectors(model, crop_pctg, split, version)
                    

    # generateVectors(
    #     model=Consts.VGG16,
    #     crop_pctg=0,
    #     split=Consts.VAL,
    #     version=Consts.ORIGINAL
    # )

if __name__ == "__main__":
    main()