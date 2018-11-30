# smallervgg16 classifer was built using this reference
# https://www.pyimagesearch.com/2018/04/16/keras-and-convolutional-neural-networks-cnns/

import keras
from keras.applications import vgg16
from keras.applications import vgg19
from keras.applications import xception
from keras.applications import densenet
from keras.applications import inception_v3

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

class bagclassifiers:
    @staticmethod
    def build_smallervgg16(height, width, depth, classes):
        model = Sequential()
        input_shape = (height, width, depth)

        # channels last since we are using tensorflow
        channel_dim = -1
        
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # since we have 4 classes (purse, carryon, dufflebag, backpack)
        # we use a softmax classifier with 4 classes
        num_classes = 4
        model.add(Dense(num_classes))
        model.add(Activation("softmax"))

        return model

    @staticmethod
    def build_vgg16(classes):
        model = vgg16.VGG16(include_top=True,
                            weights=None,
                            input_tensor=None,
                            input_shape=None,
                            pooling=None,
                            classes=classes)
        return model

    def build_vgg19(classes):
        model = vgg19.VGG19(include_top=True,
                            weights=None,
                            input_tensor=None,
                            input_shape=None,
                            pooling=None,
                            classes=classes)
        return model

    def build_xception(classes):
        model = xception.Xception(include_top=True,
                            weights=None,
                            input_tensor=None,
                            input_shape=None,
                            pooling=None,
                            classes=classes)
        return model

    def build_inception(classes):
        model = inception_v3.InceptionV3(include_top=True,
                            weights=None,
                            input_tensor=None,
                            input_shape=None,
                            pooling=None,
                            classes=classes)
        return model

