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
        model_path = Dirs.get_model_cache_dir(model)
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        crop_dirs = Dirs.get_crop_dirs(model=model)
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

def filenames_and_labels(crop_pctg, split, version):
    input_dir = Dirs.get_image_crop_split_version_dir(
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

    return zip(*filepaths_and_labels)

def vectors_and_labels(model, crop_pctg, split, version):
    base_dir = Dirs.get_vector_model_crop_split_version_dir(
        model=model,
        crop_pctg=crop_pctg,
        split=split,
        version=version
    )

    vecs_and_labels = []

    for label in os.listdir(base_dir):
        label_dir = "{}{}/".format(base_dir, label)

        for file_name in os.listdir(label_dir):
            vector_path = "{}{}".format(label_dir, file_name)
            vecs_and_labels.append((vector_path, label))

    shuffle(vecs_and_labels)

    return zip(*vecs_and_labels)

def save_vectors_parallel(iterable):
    iden_dir = "{}{}/".format(output_dir, label)
    if not os.path.exists(iden_dir):
        os.mkdir(iden_dir)

    file_name = file_path.split("/")[-1].split(".")[0]
    save_path = "{}{}.pkl".format(iden_dir, file_name)

    with open(save_path, "wb") as vector:
        pickle.dump(f_v, vector)

def generate_vectors(model, crop_pctg, split, version):
    filenames, labels = filenames_and_labels(
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

    # sequential (shitty af)
    # for f_v, file_path, label in zip(predictions, filenames, labels):
    #     iden_dir = "{}{}/".format(output_dir, label)
    #     if not os.path.exists(iden_dir):
    #         os.mkdir(iden_dir)

    #     file_name = file_path.split("/")[-1].split(".")[0]
    #     save_path = "{}{}.pkl".format(iden_dir, file_name)

    #     with open(save_path, "wb") as vector:
    #         pickle.dump(f_v, vector)

    # parallel
    iterable = zip(predictions, filenames, labels)

def main():
    generate_cache_tree(Dirs.ROOT_DIR)

    for model in Consts.MODELS[:2]: # porque ainda só tenho dois modelos
        for crop_pctg in Consts.CROP_PCTGS[1:]: #[1:] porque o no_crop merdou
            for split in Consts.SPLITS:
                for version in Consts.VERSIONS:
                    print("GENERATING VECTORS FOR {} crop_{pctg:02d} {} {}"
                          .format(
                              model,
                              split,
                              version,
                              pctg=crop_pctg
                          ))
                    generate_vectors(model, crop_pctg, split, version)

if __name__ == "__main__":
    main()

    # vectors_and_labels(Consts.VGG16, Consts.CROP_PCTGS[1], Consts.VAL, Consts.ORIGINAL)
