from __future__ import absolute_import, division, print_function

import os
import random
import pickle

from multiprocessing import Pool
from functools import partial

from load_local_model import load_local_fv
from generator import Generator
from new_utils import Dirs, Consts, Flags
from iterable import Iterable
from params import Params, Config

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

def vectors_and_labels(params, config):
    vecs_and_labels = []
    unknowns = []
    dict_by_iden = {}

    if config.split == Consts.TRAIN:

        og_dir = Dirs.get_vector_model_crop_split_version_dir(
            model=params.model,
            crop_pctg=params.crop_pctg,
            split=config.split,
            version=Consts.ORIGINAL
        )

        for label in os.listdir(og_dir):
            if int(label) < params.num_classes:
                if label not in dict_by_iden.keys():
                    dict_by_iden[label] = []

                label_dir = "{}{}/".format(og_dir, label)
                for i, vector in enumerate(os.listdir(label_dir)):
                    if i >= params.examples_per_class:
                        break

                    vector_id = vector.split(".")[0]
                    dict_by_iden[label].append(vector_id)

    for version in config.list_versions:
        if version[0]:
            base_dir = Dirs.get_vector_model_crop_split_version_dir(
                model=params.model,
                crop_pctg=params.crop_pctg,
                split=config.split,
                version=version[1]
            )

            for label in os.listdir(base_dir):
                label_dir = "{}{}/".format(base_dir, label)
                if int(label) < params.num_classes:

                    for i, file_name in enumerate(os.listdir(label_dir)):

                        if dict_by_iden:
                            vector_id = file_name.split(".")[0]

                            if vector_id in dict_by_iden[label]:
                                vector_path = "{}{}".format(label_dir, file_name)
                                vecs_and_labels.append((vector_path, label))

                        elif i < params.examples_per_class:
                            vector_path = "{}{}".format(label_dir, file_name)
                            vecs_and_labels.append((vector_path, label))

                # this condition add files that belong to the "unknown" class
                elif len(unknowns) * 10 < len(vecs_and_labels):
                    random_file = random.choice(os.listdir(label_dir))
                    vector_path = "{}{}".format(label_dir, random_file)
                    unknowns.append((vector_path, params.num_classes))

    vecs_and_labels += unknowns

    random.shuffle(vecs_and_labels)

    return zip(*vecs_and_labels)

def save_vectors_parallel(element, output_dir):
    iden_dir = "{}{}/".format(output_dir, element.label)
    if not os.path.exists(iden_dir):
        os.mkdir(iden_dir)

    file_name = element.filename.split("/")[-1].split(".")[0]
    save_path = "{}{}.pkl".format(iden_dir, file_name)

    with open(save_path, "wb") as vector:
        pickle.dump(element.prediction, vector)
        vector.close()

def generate_vectors(model, crop_pctg, split, version):
    filenames, labels = filenames_and_labels(
        crop_pctg=crop_pctg,
        split=split,
        version=version
    )

    output_dir = Dirs.get_vector_model_crop_split_version_dir(
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

    # parallel
    iterable = []
    for prediction, filename, label in zip(predictions, filenames, labels):
        iterable.append(Iterable(prediction, filename, label))
   
    pool = Pool(Consts.N_WORKERS)
    pool = Pool(1)
    func = partial(save_vectors_parallel, output_dir=output_dir)
    pool.map(func, list(iterable))
    pool.close()
    pool.join()

def main():
    generate_cache_tree(Dirs.ROOT_DIR)

    for model in Consts.MODELS[1:2]: # [:2] porque ainda só tenho dois modelos
        for crop_pctg in Consts.CROP_PCTGS[1:]: # [1:] porque o no_crop merdou
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
    # main()
    c = Config(Consts.TRAIN, True, True, True)
    p = Params(Consts.INCEPTIONV3, 10, 5, 20)
    vectors_and_labels(p, c)
