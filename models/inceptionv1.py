from .base import Inception
from tensorflow import keras


class InceptionV1(Inception):

    @classmethod
    def get_inception_v1_block(cls, _input, filter_1x1,
                               filter_3x3_reduce, filter_3x3,
                               filter_5x5_reduce, filter_5x5,
                               filter_pool, name=None):
        conv1 = keras.layers.Conv2D(filter_1x1, kernel_size=1, activation='relu', padding='same')(_input)

        conv3_reduce = keras.layers.Conv2D(filter_3x3_reduce, kernel_size=1, activation='relu', padding='same')(_input)
        conv3 = keras.layers.Conv2D(filter_3x3, kernel_size=3, activation='relu', padding='same')(conv3_reduce)

        conv5_reduce = keras.layers.Conv2D(filter_5x5_reduce, kernel_size=1, activation='relu', padding='same')(_input)
        conv5 = keras.layers.Conv2D(filter_5x5, kernel_size=5, activation='relu', padding='same')(conv5_reduce)

        maxpool = keras.layers.MaxPool2D((3, 3), strides=(1, 1), padding='same')(_input)
        maxpool = keras.layers.Conv2D(filter_pool, kernel_size=1, activation='relu', padding='same')(maxpool)

        _output = keras.layers.concatenate([conv1, conv3, conv5, maxpool], axis=3, name=name)

        return _output

    def define_model(self):
        inputs = keras.Input(shape=(self._image_height, self._image_width, 3))

        conv1 = keras.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(inputs)
        pool1 = keras.layers.MaxPool2D((3, 3), strides=(2, 2))(conv1)

        conv2 = keras.layers.Conv2D(192, kernel_size=(3, 3), strides=(1, 1))(pool1)
        pool2 = keras.layers.MaxPool2D((3, 3), strides=(2, 2))(conv2)

        inception1 = self.get_inception_v1_block(pool2, 64, 96, 128, 16, 32, 32, 'inception_3a')
        inception2 = self.get_inception_v1_block(inception1, 128, 128, 192, 32, 96, 64, 'inception_3b')

        pool3 = keras.layers.MaxPool2D((3, 3), strides=(2, 2))(inception2)

        inception3 = self.get_inception_v1_block(pool3, 192, 96, 208, 16, 48, 64, 'inception_4a')

        auxiliary_1 = Inception.get_auxiliary_output(inception3, 'auxiliary_1')

        inception4 = self.get_inception_v1_block(inception3, 160, 112, 224, 24, 64, 64, 'inception_4b')
        inception5 = self.get_inception_v1_block(inception4, 128, 128, 256, 24, 64, 64, 'inception_4c')
        inception6 = self.get_inception_v1_block(inception5, 112, 144, 288, 32, 64, 64, 'inception_4d')

        auxiliary_2 = Inception.get_auxiliary_output(inception6, 'auxiliary_2')

        inception7 = self.get_inception_v1_block(inception6, 256, 160, 320, 32, 128, 128, 'inception_4e')

        pool4 = keras.layers.MaxPool2D((3, 3), strides=(2, 2))(inception7)

        inception8 = self.get_inception_v1_block(pool4, 256, 160, 320, 32, 128, 128, 'inception_5a')
        inception9 = self.get_inception_v1_block(inception8, 384, 192, 384, 48, 128, 128, 'inception_5b')

        avg1 = keras.layers.GlobalAveragePooling2D()(inception9)
        dropout1 = keras.layers.Dropout(0.4)(avg1)

        flatten = keras.layers.Flatten()(dropout1)
        fc1 = keras.layers.Dense(1000, activation='relu')(flatten)
        logit = keras.layers.Dense(10, activation='softmax', name='output')(fc1)

        self.model = keras.Model(inputs=inputs, outputs=[logit, auxiliary_1, auxiliary_2])

    def train(self, epochs=100):
        self.model.compile(
            loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'],
            optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=False),
            loss_weights=[1.0, 0.3, 0.3],
            metrics=['accuracy']
        )

        _ = self.model.fit_generator(self.training.generate(),
                                     steps_per_epoch=self.training.steps_per_epoch(),
                                     validation_data=self.validation.generate(),
                                     validation_steps=self.validation.steps_per_epoch(),
                                     epochs=epochs)