from __future__ import absolute_import, division, print_function

import keras

from load_local_model import load_facenet_fv, load_vgg_face_fv, classifier
from train_final_layer import get_fv_len
from params import Params, HyperParams
from new_utils import Consts

base_learning_rate = 0.001
params = Params(
    model=Consts.VGG16,
    num_classes=100,
    examples_per_class=100,
    crop_pctg=20,
    include_unknown=True
)
hyper_params = HyperParams(
    num_epochs=200,
    dropout_rate=0.75,
    batch_size=64,
    final_layer_activation=keras.activations.softmax,
    optimizer=keras.optimizers.RMSprop(
        lr=base_learning_rate
    )
)

vgg_layer = load_vgg_face_fv()

prediction_layer = classifier(
    fv_len=get_fv_len(params.model),
    num_classes=params.num_classes+1,
    dropout_rate=hyper_params.dropout_rate,
    final_layer_activation=keras.activations.softmax,
    optimizer=hyper_params.optimizer
)
prediction_layer.load_weights("/home/gustavoduartefaria/gFaceRec/training/training_0481/cp-0200.hdf5")

model = keras.Sequential([
    vgg_layer,
    prediction_layer
])

model.compile(
    optimizer=keras.optimizers.Adam(
        lr=base_learning_rate
    ),
    loss=keras.losses.sparse_categorical_crossentropy,
    metrics=['accuracy']
)

model.summary()

model.save('/home/gustavoduartefaria/stacked_model_sheer.hdf5')
