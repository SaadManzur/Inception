import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from tensorflow import keras
import numpy as np
import cv2 as opencv


def get_inception_block(_input, filter_1x1,
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


def get_auxilary_output(_input, name=None):

    auxilary = keras.layers.AvgPool2D((5, 5), strides=(3, 3))(_input)
    auxilary = keras.layers.Conv2D(128, (1, 1), padding='same', activation='relu')(auxilary)
    auxilary = keras.layers.Flatten()(auxilary)
    auxilary = keras.layers.Dense(1024, activation='relu')(auxilary)
    auxilary = keras.layers.Dropout(0.7)(auxilary)
    auxilary = keras.layers.Dense(10, activation='softmax', name=name)(auxilary)

    return auxilary


def GoogLeNet():
    inputs = keras.Input(shape=(224, 224, 3))

    conv1 = keras.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(inputs)
    pool1 = keras.layers.MaxPool2D((3, 3), strides=(2, 2))(conv1)

    conv2 = keras.layers.Conv2D(192, kernel_size=(3, 3), strides=(1, 1))(pool1)
    pool2 = keras.layers.MaxPool2D((3, 3), strides=(2, 2))(conv2)

    inception1 = get_inception_block(pool2, 64, 96, 128, 16, 32, 32, 'inception_3a')
    inception2 = get_inception_block(inception1, 128, 128, 192, 32, 96, 64, 'inception_3b')

    pool3 = keras.layers.MaxPool2D((3, 3), strides=(2, 2))(inception2)

    inception3 = get_inception_block(pool3, 192, 96, 208, 16, 48, 64, 'inception_4a')

    auxilary1 = get_auxilary_output(inception3, 'auxilary_1')

    inception4 = get_inception_block(inception3, 160, 112, 224, 24, 64, 64, 'inception_4b')
    inception5 = get_inception_block(inception4, 128, 128, 256, 24, 64, 64, 'inception_4c')
    inception6 = get_inception_block(inception5, 112, 144, 288, 32, 64, 64, 'inception_4d')

    auxilary2 = get_auxilary_output(inception6, 'auxilary_2')

    inception7 = get_inception_block(inception6, 256, 160, 320, 32, 128, 128, 'inception_4e')

    pool4 = keras.layers.MaxPool2D((3, 3), strides=(2, 2))(inception7)

    inception8 = get_inception_block(pool4, 256, 160, 320, 32, 128, 128, 'inception_5a')
    inception9 = get_inception_block(inception8, 384, 192, 384, 48, 128, 128, 'inception_5b')

    avg1 = keras.layers.GlobalAveragePooling2D()(inception9)
    dropout1 = keras.layers.Dropout(0.4)(avg1)

    flatten = keras.layers.Flatten()(dropout1)
    fc1 = keras.layers.Dense(1000, activation='relu')(flatten)
    logit = keras.layers.Dense(10, activation='softmax', name='output')(fc1)

    model = keras.Model(inputs=inputs, outputs=[logit, auxilary1, auxilary2])

    return model


def get_cifar10_dataset(width, height):
    (x_train, y_train), (x_valid, y_valid) = keras.datasets.cifar10.load_data()

    x_train = np.array([opencv.resize(img, (width, height)) for img in x_train[:, :, :, :]])
    x_valid = np.array([opencv.resize(img, (width, height)) for img in x_valid[:, :, :, :]])

    y_train = keras.utils.to_categorical(y_train, 10)
    y_valid = keras.utils.to_categorical(y_valid, 10)

    print(y_train.shape)
   
    print("Changing types")
    x_train = x_train.astype('float32')
    x_valid = x_valid.astype('float32')

    print("Scaling")
    x_train /= 255.0
    x_valid /= 255.0

    return x_train, y_train, x_valid, y_valid


def multiple_outputs_generator(generator, x, y, batch_size=256):

    generator_x = generator.flow(x, y, batch_size=batch_size)

    while True:
        x_i, y_i = generator_x.next()

        yield x_i, {'output_1': y_i, 'output_2': y_i, 'output_3': y_i}


def train_with_cifar10():
    x_train, y_train, x_valid, y_valid = get_cifar10_dataset(224, 224)

    model = GoogLeNet()

    model.compile(
        loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'],
        optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=False),
        loss_weights=[1.0, 0.3, 0.3],
        metrics=['accuracy']
    )

    data_generator = keras.preprocessing.image.ImageDataGenerator()

    _ = model.fit_generator(multiple_outputs_generator(data_generator, x_train, y_train), steps_per_epoch=50000//256,
                            validation_data=multiple_outputs_generator(data_generator, x_valid, y_valid),
                            validation_steps=10000//256, epochs=100)


if __name__ == '__main__':
    train_with_cifar10()
