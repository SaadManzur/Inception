from .base import Inception
from tensorflow import keras


class InceptionV3(Inception):

    def get_inception_a(self, _input, filter_1x1, filter_3x3_reduce, filter_3x3,
                        filter_5x5_reduce, filter_5x5, filter_pool, name="inception"):
        name += '_a'

        conv_1x1 = keras.layers.Conv2D(filter_1x1, kernel_size=1, activation='relu',
                                       padding='same', name=name+'_1x1')(_input)

        pool = keras.layers.AvgPool2D((3, 3), strides=(1, 1),padding='same', name=name+'_avg_3x3')(_input)
        pool_reduce = keras.layers.Conv2D(filter_pool, kernel_size=1, padding='same',
                                          activation='relu', name=name+'_reduce_1')(pool)

        reduce_3x3 = keras.layers.Conv2D(filter_3x3_reduce, kernel_size=1, activation='relu',
                                         padding='same', name=name+'_reduce_2')(_input)
        conv_3x3 = keras.layers.Conv2D(filter_3x3, kernel_size=3, activation='relu',
                                       padding='same', name=name+'_3x3')(reduce_3x3)

        reduce_5x5 = keras.layers.Conv2D(filter_5x5_reduce, kernel_size=1, activation='relu',
                                         padding='same', name=name+'_reduce_3')(_input)
        conv_3x3_a = keras.layers.Conv2D(filter_5x5, kernel_size=3, activation='relu',
                                         padding='same', name=name+'_3x3_1')(reduce_5x5)
        conv_3x3_b = keras.layers.Conv2D(filter_5x5, kernel_size=3, activation='relu',
                                         padding='same', name=name+'_3x3_2')(conv_3x3_a)

        _output = keras.layers.concatenate([conv_3x3_b, conv_3x3, pool_reduce, conv_1x1], axis=3, name=name)

        return _output

    def get_inception_b(self, _input, filter_1x1, filter_7x7_a_reduce, filter_7x7_a,
                        filter_7x7_b_reduce, filter_7x7_b, filter_pool, name='inception'):
        name += '_b'

        conv_1x1 = keras.layers.Conv2D(filter_1x1, kernel_size=1, activation='relu',
                                       padding='same', name=name+'_1x1')(_input)

        pool = keras.layers.AvgPool2D((3, 3), strides=(1, 1), padding='same', name=name+'_avg_3x3')(_input)
        pool_reduce = keras.layers.Conv2D(filter_pool, kernel_size=1, padding='same',
                                          activation='relu', name=name+'_reduce_1')(pool)

        reduce_7x7 = keras.layers.Conv2D(filter_7x7_a_reduce, kernel_size=7, activation='relu',
                                         padding='same', name=name+'_reduce_2')(_input)
        conv_1x7_a = keras.layers.Conv2D(filter_7x7_a, kernel_size=(1, 7), activation='relu',
                                         padding='same', name=name+'_1x7_a')(reduce_7x7)
        conv_7x1_a = keras.layers.Conv2D(filter_7x7_a, kernel_size=(7, 1), activation='relu',
                                         padding='same', name=name+'_7x1_a')(conv_1x7_a)

        reduce_7x7 = keras.layers.Conv2D(filter_7x7_b_reduce, kernel_size=7, activation='relu',
                                         padding='same', name=name+'_reduce_3')(_input)
        conv_7x1_b = keras.layers.Conv2D(filter_7x7_b, kernel_size=(7, 1), activation='relu',
                                         padding='same', name=name+'_7x1_b1')(reduce_7x7)
        conv_1x7_b = keras.layers.Conv2D(filter_7x7_b, kernel_size=(1, 7), activation='relu',
                                         padding='same', name=name+'_1x7_b1')(conv_7x1_b)
        conv_7x1_b = keras.layers.Conv2D(filter_7x7_b, kernel_size=(7, 1), activation='relu',
                                         padding='same', name=name+'_7x1_b2')(conv_1x7_b)
        conv_1x7_b = keras.layers.Conv2D(filter_7x7_b, kernel_size=(1, 7), activation='relu',
                                         padding='same', name=name+'_1x7_b2')(conv_7x1_b)

        _output = keras.layers.concatenate([conv_1x7_b, conv_7x1_a, pool_reduce, conv_1x1], axis=3, name=name)

        return _output

    def get_inception_c(self, _input, filter_1x1, filter_3x3_a_reduce, filter_3x3_a,
                        filter_3x3_b_reduce, filter_3x3_b, filter_pool, name='inception'):
        name += '_c'

        conv_1x1 = keras.layers.Conv2D(filter_1x1, kernel_size=1, activation='relu',
                                       padding='same', name=name+'_1x1')(_input)

        pool = keras.layers.AvgPool2D((3, 3), strides=(1, 1), padding='same', name=name+'_avg_3x3')(_input)
        pool_reduce = keras.layers.Conv2D(filter_pool, kernel_size=1, padding='same',
                                          activation='relu', name=name+'_reduce_1')(pool)

        reduce_3x3 = keras.layers.Conv2D(filter_3x3_a_reduce, kernel_size=1, padding='same',
                                         activation='relu', name=name+'_reduce_2')(_input)
        conv_1x3_a = keras.layers.Conv2D(filter_3x3_a, kernel_size=(1, 3), padding='same',
                                         activation='relu', name=name+'_1x3_a')(reduce_3x3)
        conv_3x1_a = keras.layers.Conv2D(filter_3x3_a, kernel_size=(3, 1), padding='same',
                                         activation='relu', name=name+'_3x1_a')(reduce_3x3)

        reduce_3x3 = keras.layers.Conv2D(filter_3x3_b_reduce, kernel_size=1, padding='same',
                                         activation='relu', name=name+'_reduce_3')(_input)
        conv_3x3 = keras.layers.Conv2D(filter_3x3_b, kernel_size=3, padding='same',
                                       activation='relu', name=name+'_3x3')(reduce_3x3)
        conv_3x1_b = keras.layers.Conv2D(filter_3x3_b, kernel_size=(3, 1), padding='same',
                                         activation='relu', name=name+'_3x1_b')(conv_3x3)
        conv_1x3_b = keras.layers.Conv2D(filter_3x3_b, kernel_size=(1, 3), padding='same',
                                         activation='relu', name=name+'_1x3_b')(conv_3x3)

        _output = keras.layers.concatenate([conv_3x1_b, conv_1x3_b, conv_1x3_a, conv_3x1_a, pool_reduce, conv_1x1],
                                           axis=3, name=name)

        return _output

    def grid_size_reduction_a(self, _input, filter_1x1, filter_3x3_a, filter_3x3_b, name='reduction'):
        conv_1x1 = keras.layers.Conv2D(filter_1x1, kernel_size=1, activation='relu',
                                       padding='same', name=name+'_1x1')(_input)
        conv_3x3_a = keras.layers.Conv2D(filter_3x3_a, kernel_size=3, activation='relu',
                                         padding='same', name=name+'_3x3_a1')(conv_1x1)
        conv_3x3_a = keras.layers.Conv2D(filter_3x3_a, kernel_size=3, activation='relu',
                                         padding='same', name=name+'_3x3_a2')(conv_3x3_a)

        conv_3x3_b = keras.layers.Conv2D(filter_3x3_b, kernel_size=3, activation='relu',
                                         padding='same', name=name+'_3x3_b')(_input)

        maxpool = keras.layers.MaxPool2D((3, 3), strides=(1, 1), padding='same', name=name+'_max_3x3')(_input)

        _output = keras.layers.concatenate([conv_3x3_a, conv_3x3_b, maxpool], axis=3, name=name + '_a')

        return _output

    def grid_size_reduction_b(self, _input, filter_7x7_reduce, filter_7x7, filter_3x3_a,
                              filter_3x3_reduce, filter_3x3_b, name='reduction'):
        reduce_7x7 = keras.layers.Conv2D(filter_7x7_reduce, kernel_size=1, activation='relu',
                                         padding='same', name=name+'_reduce_1')(_input)
        conv_1x7 = keras.layers.Conv2D(filter_7x7, kernel_size=(1, 7), activation='relu',
                                       padding='same', name=name+'_1x7')(reduce_7x7)
        conv_7x1 = keras.layers.Conv2D(filter_7x7, kernel_size=(7, 1), activation='relu',
                                       padding='same', name=name+'_7x1')(conv_1x7)
        conv_3x3_a = keras.layers.Conv2D(filter_3x3_a, kernel_size=3, activation='relu',
                                         padding='same', name=name+'_3x3_a')(conv_7x1)

        reduce_3x3 = keras.layers.Conv2D(filter_3x3_reduce, kernel_size=1, activation='relu',
                                         padding='same', name=name+'_reduce_2')(_input)
        conv_3x3_b = keras.layers.Conv2D(filter_3x3_b, kernel_size=3, activation='relu',
                                         padding='same', name=name+'_3x3_b')(reduce_3x3)

        maxpool = keras.layers.MaxPool2D((3, 3), strides=(1, 1), padding='same', name=name+'_max_3x3')(_input)

        _output = keras.layers.concatenate([conv_3x3_a, conv_3x3_b, maxpool], axis=3, name=name + '_b')

        return _output

    def get_stem(self, _input):
        conv_3x3 = keras.layers.Conv2D(32, kernel_size=3, activation='relu', name='stem_3x3_1')(_input)
        conv_3x3 = keras.layers.Conv2D(32, kernel_size=3, activation='relu', name='stem_3x3_2')(conv_3x3)
        conv_3x3 = keras.layers.Conv2D(32, kernel_size=3, activation='relu', name='stem_3x3_3')(conv_3x3)

        maxpool = keras.layers.MaxPool2D((3, 3), strides=(1, 1), padding='same', name='stem_max_3x3_1')(conv_3x3)

        conv_1x1 = keras.layers.Conv2D(32, kernel_size=1, activation='relu', name='stem_1x1_1')(maxpool)

        conv_3x3 = keras.layers.Conv2D(32, kernel_size=3, activation='relu', name='stem_3x3_4')(conv_1x1)

        maxpool = keras.layers.MaxPool2D((3, 3), strides=(1, 1), padding='same', name='stem_max_3x3_2')(conv_3x3)

        return maxpool

    def get_auxiliary_output(self, _input, name=None):
        auxiliary = keras.layers.AvgPool2D((5, 5), strides=(3, 3))(_input)
        auxiliary = keras.layers.Conv2D(128, (1, 1), padding='same', activation='relu')(auxiliary)
        auxiliary = keras.layers.BatchNormalization()(auxiliary)
        auxiliary = keras.layers.Flatten()(auxiliary)
        auxiliary = keras.layers.Dense(768, activation='relu')(auxiliary)
        auxiliary = keras.layers.BatchNormalization()(auxiliary)
        auxiliary = keras.layers.Dense(self.num_of_classes, activation='softmax', name=name)(auxiliary)

        return auxiliary

    def define_model(self):
        inputs = keras.Input(shape=(self._image_height, self._image_width, 3))

        stem = self.get_stem(inputs)

        inception_a = self.get_inception_a(stem, 32, 32, 32, 32, 32, 32, '1')
        inception_a = self.get_inception_a(inception_a, 32, 32, 32, 32, 32, 32, '2')
        inception_a = self.get_inception_a(inception_a, 32, 32, 32, 32, 32, 32, '3')

        reduction_a = self.grid_size_reduction_a(inception_a, 32, 32, 32, 'red1')

        inception_b = self.get_inception_b(reduction_a, 32, 32, 32, 32, 32, 32, '1')
        inception_b = self.get_inception_b(inception_b, 32, 32, 32, 32, 32, 32, '2')
        inception_b = self.get_inception_b(inception_b, 32, 32, 32, 32, 32, 32, '3')
        inception_b = self.get_inception_b(inception_b, 32, 32, 32, 32, 32, 32, '4')

        auxiliary_1 = self.get_auxiliary_output(inception_b, 'auxiliary_1')

        reduction_b = self.grid_size_reduction_b(inception_b, 32, 32, 32, 32, 32, 'red2')

        inception_c = self.get_inception_c(reduction_b, 32, 32, 32, 32, 32, 32, '1')
        inception_c = self.get_inception_c(inception_c, 32, 32, 32, 32, 32, 32, '2')

        global_avg_pool = keras.layers.GlobalAveragePooling2D()(inception_c)

        fc1 = keras.layers.Dense(2048, activation='relu')(global_avg_pool)
        output = keras.layers.Dense(self.num_of_classes, activation='softmax')(fc1)

        self.model = keras.Model(inputs=inputs, outputs=[output, auxiliary_1])

    def train(self, epochs=100):
        self.model.compile(
            loss=['categorical_crossentropy', 'categorical_crossentropy'],
            optimizer=keras.optimizers.RMSProp(learning_rate=0.01, momentum=0.9),
            loss_weights=[1.0, 0.3],
            metrics=['accuracy']
        )

        _ = self.model.fit_generator(self.training.generate(),
                                     steps_per_epoch=self.training.steps_per_epoch(),
                                     validation_data=self.validation.generate(),
                                     validation_steps=self.validation.steps_per_epoch(),
                                     epochs=epochs)





