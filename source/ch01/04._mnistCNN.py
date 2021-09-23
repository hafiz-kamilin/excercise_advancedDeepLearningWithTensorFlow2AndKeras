#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convoluted neural network (CNN) implementation
on solving the MNIST dataset classification.

"""

# to supress tensorflow-gpu debug information
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.datasets import mnist

# load mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# compute the number of labels
num_labels = len(np.unique(y_train))

# convert to one-hot vector
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# input image dimensions
image_size = x_train.shape[1]
# resize and normalize
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# network parameters
# image is processed as is (square grayscale)
input_shape = (image_size, image_size, 1)
batch_size = 128
kernel_size = 3
pool_size = 2
filters = 64
dropout = 0.2

# model is a stack of CNN-ReLU-MaxPooling
model = Sequential()
model.add(
    Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        activation='relu',
        input_shape=input_shape
    )
)
model.add(
    MaxPooling2D(
        pool_size
    )
)
model.add(
    Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        activation='relu'
    )
)
model.add(
    MaxPooling2D(
        pool_size
    )
)
model.add(
    Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        activation='relu'
    )
)
model.add(
    Flatten()
)
# dropout added as regularizer
model.add(
    Dropout(
        dropout
    )
)
# output layer is 10-dim one-hot vector
model.add(
    Dense(
        num_labels
    )
)
model.add(
    Activation(
        'softmax'
    )
)

# shows the model summary and save it
model.summary()
plot_model(model, to_file='cnn-mnist.png', show_shapes=True)

# loss function for one-hot vector
# use of adam optimizer
# accuracy is good metric for classification tasks
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# train the network
model.fit(x_train, y_train, epochs=10, batch_size=batch_size)
_, acc = model.evaluate(
    x_test,
    y_test,
    batch_size=batch_size,
    verbose=0
)

#  shows the accuracy
print("\nTest accuracy: %.2f%%" % (100.0 * acc))