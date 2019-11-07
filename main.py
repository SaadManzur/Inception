from tensorflow import keras
import numpy as np
import cv2 as opencv


def get_inception_block(_input, filter_1x1, filter_3x3_reduce, filter_3x3, filter_5x5_reduce, filter_5x5, filter_pool):
    conv1 = keras.layers.Conv2D(filter_1x1, kernel_size=1, activation='relu', padding='same')(_input)

    conv3_reduce = keras.layers.Conv2D(filter_3x3_reduce, kernel_size=1, activation='relu', padding='same')(_input)
    conv3 = keras.layers.Conv2D(filter_3x3, kernel_size=3, activation='relu', padding='same')(conv3_reduce)

    conv5_reduce = keras.layers.Conv2D(filter_5x5_reduce, kernel_size=1, activation='relu', padding='same')(_input)
    conv5 = keras.layers.Conv2D(filter_5x5, kernel_size=5, activation='relu', padding='same')(conv5_reduce)

    maxpool = keras.layers.MaxPool2D((3, 3), strides=(1, 1), padding='same')(_input)
    maxpool = keras.layers.Conv2D(filter_pool, kernel_size=1, activation='relu', padding='same')(maxpool)

    _output = keras.layers.concatenate([conv1, conv3, conv5, maxpool], axis=3)

    return _output


def build_model():
    inputs = keras.Input(shape=(224, 224, 3))

    conv1 = keras.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(inputs)
    pool1 = keras.layers.MaxPool2D((3, 3), strides=(2, 2))(conv1)

    conv2 = keras.layers.Conv2D(192, kernel_size=(3, 3), strides=(1, 1))(pool1)
    pool2 = keras.layers.MaxPool2D((3, 3), strides=(2, 2))(conv2)

    inception1 = get_inception_block(pool2, 64, 96, 128, 16, 32, 32)
    inception2 = get_inception_block(inception1, 128, 128, 192, 32, 96, 64)

    pool3 = keras.layers.MaxPool2D((3, 3), strides=(2, 2))(inception2)

    inception3 = get_inception_block(pool3, 192, 96, 208, 16, 48, 64)
    inception4 = get_inception_block(inception3, 160, 112, 224, 24, 64, 64)
    inception5 = get_inception_block(inception4, 128, 128, 256, 24, 64, 64)
    inception6 = get_inception_block(inception5, 112, 144, 288, 32, 64, 64)
    inception7 = get_inception_block(inception6, 256, 160, 320, 32, 128, 128)

    pool4 = keras.layers.MaxPool2D((3, 3), strides=(2, 2))(inception7)

    inception8 = get_inception_block(inception7, 256, 160, 320, 32, 128, 128)
    inception9 = get_inception_block(inception8, 384, 192, 384, 48, 128, 128)

    avg1 = keras.layers.AvgPool2D((7, 7), strides=(1, 1))(inception9)
    dropout1 = keras.layers.Dropout(0.4)(avg1)

    flatten = keras.layers.Flatten()(dropout1)
    fc1 = keras.layers.Dense(1000, activation='relu')(flatten)
    logit = keras.layers.Dense(10, activation='softmax')(fc1)

    model = keras.Model(inputs=inputs, outputs=logit)

    return model


def get_cifar10_dataset(width, height):
    (x_train, y_train), (x_valid, y_valid) = keras.datasets.cifar10.load_data()

    x_train = np.array([opencv.resize(img, (width, height)) for img in x_train[:, :, :, :]])
    x_valid = np.array([opencv.resize(img, (width, height)) for img in x_valid[:, :, :, :]])

    y_train = keras.utils.to_categorical(y_train, 10)
    y_valid = keras.utils.to_categorical(y_valid, 10)
    
    print("Changing types")
    x_train = x_train.astype('float32')
    x_valid = x_valid.astype('float32')

    print("Scaling")
    x_train /= 255.0
    x_valid /= 255.0

    return x_train, y_train, x_valid, y_valid


if __name__ == '__main__':
    x_train, y_train, x_valid, y_valid = get_cifar10_dataset(224, 224)

    model = build_model()

    model.compile(
        loss=['categorical_crossentropy'],
        optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=False),
        metrics=['accuracy']
    )

    augment = keras.preprocessing.image.ImageDataGenerator()

    _ = model.fit_generator(augment.flow(x_train, y_train, 256),
                            validation_data=(x_valid, y_valid), epochs=100)
