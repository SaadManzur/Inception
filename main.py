import os
from tensorflow import keras
import numpy as np
import cv2 as opencv
from models.inceptionv1 import InceptionV1
from models.inceptionv3 import InceptionV3

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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


if __name__ == '__main__':
    x_train, y_train, x_valid, y_valid = get_cifar10_dataset(224, 224)

    model = InceptionV3([224, 224])
    model.define_model()
    model.plot_model()
    model.set_training_data(x_train, y_train, 32, 3)
    model.set_validation_data(x_valid, y_valid, 256, 3)
    model.smooth_labels(smooth_factor=0.1)
    model.train(100)
