from __future__ import absolute_import, division, print_function

from keras_vggface.vggface import VGGFace
import keras
from keras.models import load_model

# import tensorflow_hub as hub
from new_utils import Consts

def load_face_detector():
    model = load_model("detector/yolov2_tiny-face.h5")
    model.trainable = False

    return model

def load_vgg_face_fv():
    vgg_model = VGGFace(model='vgg16', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    vgg_model.trainable = False

    return vgg_model

def load_facenet_fv():
    model = load_model("facenet/model.h5")
    model.load_weights("facenet/weights.h5")
    model.trainable = False

    return model

def load_local_fv(model):
    """Chooses which of the pretrained models to return
    
    Arguments:
        model {string} -- Chosen model
    
    Returns:
        keras model -- CNN
    """
    if model == Consts.INCEPTIONV3:
        return load_facenet_fv()
    elif model == Consts.VGG16:
        return load_vgg_face_fv()
    else:
        print("Not implemented yet")
        return None
        # TODO: resnet e senet

def classifier(fv_len, num_classes, dropout_rate, final_layer_activation, optimizer):
    """Small classifier with a feature vector as input and class scores as output

    Arguments:
        fv_len {integer} -- 128 if InceptionV3, 512 otherwise
        num_classes {integer} -- amount of discrete classes
        final_layer_activation {keras.activations.*}

    Keyword Arguments:
        dropout_rate {float} -- float value between 0 and 1 (default: {Consts.DROPOUT_RATE})

    Returns:
        keras model
    """

    model = keras.models.Sequential([
        keras.layers.Dense(
            units=fv_len,
            activation=keras.activations.relu,
            input_shape=(fv_len,),
        ),
        keras.layers.Dropout(
            rate=dropout_rate,
        ),
        keras.layers.Dense(
            units=num_classes,
            activation=final_layer_activation, # keras.activations.softmax,
            kernel_initializer=keras.initializers.he_uniform(seed=None),
        )
    ])

    model.compile(
        optimizer=optimizer, # keras.optimizers.rmsprop,
        loss=keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy'],
    )

    return model
