from pytest import fixture

import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import MaxPool2D, ReLU, Conv2D, Softmax, Dense, Flatten
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam

@fixture(scope='session', name='keras_model')
def keras_model():

    input_shape = (28,28,1)
    no_classes = 10

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(no_classes, activation='softmax'))    


    model.compile(
        loss=sparse_categorical_crossentropy,
        optimizer=Adam(),
        metrics=['accuracy']
    )

    np.random.seed(12345)
    mu, sigma = 0, 0.1 # mean and standard deviation
    x = np.random.normal(mu, sigma, size = (1,) + input_shape)
    y = model(x)

    return model

@fixture(scope='session', name='keras_model_dset')
def keras_model_dset():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    num_calibration_steps = 128
    calibration_dtype = tf.float32
    input_shape = (28,28,1)

    def representative_dataset_gen():
        for _ in range(num_calibration_steps):
            rand_idx = np.random.randint(0, x_test.shape[0]-1)
            sample = x_test[rand_idx]
            sample = sample[tf.newaxis, ...]
            sample = tf.cast(sample, dtype=calibration_dtype)
            yield [sample]

    return representative_dataset_gen