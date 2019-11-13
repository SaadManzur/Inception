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

    @classmethod
    def get_auxiliary_output(cls, _input, name=None):
        auxiliary = keras.layers.AvgPool2D((5, 5), strides=(3, 3))(_input)
        auxiliary = keras.layers.Conv2D(128, (1, 1), padding='same', activation='relu')(auxiliary)
        auxiliary = keras.layers.Flatten()(auxiliary)
        auxiliary = keras.layers.Dense(1024, activation='relu')(auxiliary)
        auxiliary = keras.layers.Dropout(0.7)(auxiliary)
        auxiliary = keras.layers.Dense(10, activation='softmax', name=name)(auxiliary)

        return auxiliary

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



