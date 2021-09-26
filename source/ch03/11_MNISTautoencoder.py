#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example of autoencoder model on MNIST dataset

This autoencoder has modular design. The encoder, decoder and autoencoder
are 3 models that share weights. For example, after training the
autoencoder, the encoder can be used to  generate latent vectors
of input data for low-dim visualization like PCA or TSNE.

"""

######################
# required libraries #
######################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# to supress tensorflow-gpu debug information
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K

import numpy as np
import matplotlib.pyplot as plt

##############
# parameters #
##############

# network parameters
batch_size = 32
kernel_size = 3
latent_dim = 16

# number of epochs
# default value is 1
epochs = 3

# encoder/decoder number of CNN layers and filters per layer
layer_filters = [32, 64]

#######################
# autoencoder builder #
#######################

def autoEncoderBuilder(input_shape):

    ##################
    # encoder module #
    ##################

    # first build the encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs

    # stack of Conv2D(32)-Conv2D(64)
    for filters in layer_filters:
        x = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=2,
            activation='relu',
            padding='same'
        )(x)

    # shape info needed to build decoder model so we don't do hand computation
    # the input to the decoder's first Conv2DTranspose will have this shape
    # shape is (7, 7, 64) which can be processed by the decoder back to (28, 28, 1)
    shape = K.int_shape(x)

    # generate the latent vector
    x = Flatten()(x)
    latent = Dense(latent_dim, name='latent_vector')(x)

    # instantiate encoder model
    encoder = Model(inputs, latent, name='encoder')
    encoder.summary()
    print("\n")
    # plot encoder
    plot_model(encoder, to_file="encoderModel.png", show_shapes=True)

    ##################
    # decoder module #
    ##################

    # build the decoder model
    latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
    # use the shape (7, 7, 64) that was earlier saved
    x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
    # from vector to suitable shape for transposed conv
    x = Reshape((shape[1], shape[2], shape[3]))(x)

    # stack of Conv2DTranspose(64)-Conv2DTranspose(32)
    for filters in layer_filters[::-1]:
        x = Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            strides=2,
            activation='relu',
            padding='same'
        )(x)

    # reconstruct the denoised input
    outputs = Conv2DTranspose(
        filters=1,
        kernel_size=kernel_size,
        padding='same',
        activation='sigmoid',
        name='decoder_output'
    )(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    print("\n")
    # plot dencoder
    plot_model(encoder, to_file="decoder.png", show_shapes=True)

    #####################################################
    # combine the encode module with the decoder module #
    ####################################################

    # autoencoder = encoder + decoder
    # instantiate autoencoder model
    autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
    autoencoder.summary()
    print("\n")

    # plot autoencoder graph
    plot_model(autoencoder, to_file="autoEncodeModel.png", show_shapes=True)

    return autoencoder

########
# main #
########

if __name__ == "__main__":

    # load MNIST dataset
    (x_train, _), (x_test, _) = mnist.load_data()

    # reshape to (28, 28, 1) and normalize input images
    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    # get the image size
    input_shape = (image_size, image_size, 1)

    # create the autoencoder model
    model = autoEncoderBuilder(input_shape)

    # compile the model with Mean Square Error (MSE) loss function and Adam as the optimizer
    model.compile(loss='mse', optimizer='adam')

    # train the autoencoder
    model.fit(
        x_train,
        x_train,
        validation_data=(x_test, x_test),
        epochs=1,
        batch_size=batch_size
    )

    # predict the autoencoder output from test data
    x_decoded = model.predict(x_test)

    # display the 1st 8 test input and decoded images
    imgs = np.concatenate([x_test[:8], x_decoded[:8]])
    imgs = imgs.reshape((4, 4, image_size, image_size))
    imgs = np.vstack([np.hstack(i) for i in imgs])
    plt.figure()
    plt.axis('off')
    plt.title('Input: 1st 2 rows, Decoded: last 2 rows')
    plt.imshow(imgs, interpolation='none', cmap='gray')
    plt.savefig('input_and_decoded.png')
    plt.show()
