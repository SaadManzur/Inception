from tensorflow import keras
from .data import Data


class Inception(object):

    def __init__(self, input_shape):
        self._image_width = input_shape[0]
        self._image_height = input_shape[1]
        self.training = None
        self.validation = None
        self.test = None
        self.model = None
        self.num_of_classes = 0

    def get_auxiliary_output(self, _input, name=None):
        auxiliary = keras.layers.AvgPool2D((5, 5), strides=(3, 3))(_input)
        auxiliary = keras.layers.Conv2D(128, (1, 1), padding='same', activation='relu')(auxiliary)
        auxiliary = keras.layers.Flatten()(auxiliary)
        auxiliary = keras.layers.Dense(1024, activation='relu')(auxiliary)
        auxiliary = keras.layers.Dropout(0.7)(auxiliary)
        auxiliary = keras.layers.Dense(self.num_of_classes, activation='softmax', name=name)(auxiliary)

        return auxiliary

    def get_stem(self, _input):
        conv1 = keras.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), activation='relu', name='stem_7x7')(_input)
        pool1 = keras.layers.MaxPool2D((3, 3), strides=(2, 2), name='stem_max_3x3_1')(conv1)

        conv2 = keras.layers.Conv2D(192, kernel_size=(3, 3), strides=(1, 1), activation='relu', name='stem_3x3')(pool1)
        pool2 = keras.layers.MaxPool2D((3, 3), strides=(2, 2), name='stem_max_3x3_2')(conv2)

        return pool2

    def define_model(self):
        return NotImplementedError

    def train(self):
        return NotImplementedError

    def set_training_data(self, x, y, batch_size, outputs=1):
        self.training = Data(x, y, x.shape[0], batch_size=batch_size, outputs=outputs)

    def set_validation_data(self, x, y, batch_size, outputs=1):
        self.validation = Data(x, y, x.shape[0], batch_size=batch_size, outputs=outputs)

    def set_test_data(self, x, y, batch_size, outputs=1):
        self.test = Data(x, y, x.shape[0], batch_size=batch_size, outputs=outputs)

    def set_num_of_classes(self, num_of_classes):
        self.num_of_classes = num_of_classes

    def plot_model(self):
        assert self.model is not None

        keras.utils.plot_model(self.model, to_file='model.png')

    def smooth_labels(self, smooth_factor):

        assert self.training is not None
        self.training.smooth_labels(smooth_factor)

        assert self.validation is not None
        self.validation.smooth_labels(smooth_factor)

        assert self.test is not None
        self.test.smooth_labels(smooth_factor)



