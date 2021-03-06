#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Y-network CNN implementation with reduced number of layer
to shorten the training time.

"""

# to supress tensorflow-gpu debug information
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model

# load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# from sparse label to categorical
num_labels = len(np.unique(y_train))
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# reshape and normalize input images
image_size = x_train.shape[1]
x_train = np.reshape(x_train,[-1, image_size, image_size, 1])
x_test = np.reshape(x_test,[-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# network parameters
input_shape = (image_size, image_size, 1)
batch_size = 32
kernel_size = 3
dropout = 0.4
n_filters = 32

# left branch of Y network
left_inputs = Input(shape=input_shape)
x = left_inputs
filters = n_filters
# 2 layers of Conv2D-Dropout-MaxPooling2D
# number of filters doubles after each layer (32-64)
for i in range(2):
    x = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        padding='same',
        activation='relu'
    )(x)
    x = Dropout(dropout)(x)
    x = MaxPooling2D()(x)
    filters *= 2

# right branch of Y network
right_inputs = Input(shape=input_shape)
y = right_inputs
filters = n_filters
# 2 layers of Conv2D-Dropout-MaxPooling2Do
# number of filters doubles after each layer (32-64)
for i in range(2):
    y = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        padding='same',
        activation='relu',
        dilation_rate=2
    )(y)
    y = Dropout(dropout)(y)
    y = MaxPooling2D()(y)
    filters *= 2

# merge left and right branches outputs
y = concatenate([x, y])
# feature maps to vector before connecting to Dense
y = Flatten()(y)
y = Dropout(dropout)(y)
outputs = Dense(
    num_labels,
    activation='softmax'
)(y)

# build the model in functional API
model = Model(
    [left_inputs, right_inputs],
    outputs
)

# verify the model using layer text description
model.summary()
plot_model(model, to_file='modified-y-network-cnn.png', show_shapes=True)

# classifier loss, Adam optimizer, classifier accuracy
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# train the model with input images and labels
model.fit(
    [x_train, x_train],
    y_train,
    validation_data=([x_test, x_test], y_test),
    epochs=20,
    batch_size=batch_size
)

# model accuracy on test dataset
score = model.evaluate(
    [x_test, x_test],
    y_test,
    batch_size=batch_size,
    verbose=0
)

print("\nTest accuracy: %.2f%%" % (100.0 * score[1]))