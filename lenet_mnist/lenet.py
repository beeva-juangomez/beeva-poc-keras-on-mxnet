from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense


class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        model.add(Convolution2D(20, 5, 5, border_mode="same",
                                input_shape=(depth, height, width)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Convolution2D(50, 5, 5, border_mode="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model

    @staticmethod
    def build_optimized(width, height, depth, classes):
        model = Sequential()
        model.add(Convolution2D(20, 5, 5, border_mode="same",
                                input_shape=(depth, height, width)))
        model.add(Activation("tanh"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Convolution2D(50, 5, 5, border_mode="same"))
        model.add(Activation("tanh"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("tanh"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model

    @staticmethod
    def build_optimized_tensorflow(width, height, depth, classes):
        model = Sequential()
        model.add(Convolution2D(20, 5, 5, border_mode="same",
                                input_shape=(depth, height, width)))
        model.add(Activation("tanh"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering="th"))
        model.add(Convolution2D(50, 5, 5, border_mode="same"))
        model.add(Activation("tanh"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering="th"))
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("tanh"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model
