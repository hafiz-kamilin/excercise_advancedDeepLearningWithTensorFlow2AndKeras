#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Autoencoder + classifier

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

from tensorflow.keras.layers import Activation, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Reshape, Conv2DTranspose, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import backend as K

import numpy as np
import matplotlib.pyplot as plt

##############
# parameters #
##############

# network parameters
batch_size = 128
kernel_size = 3
pool_size = 2
dropout = 0.4
filters = 16
latent_dim = 16

# number of epochs
# default value is 2
epochs = 2

########################################
# autoencoder + classification builder #
########################################

def AEclassification(input_shape, filters):

    ##################
    # encoder module #
    ##################

    # first build the encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs

    # stack of BN-ReLU-Conv2D-MaxPooling blocks
    for _ in range(2):
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        filters = filters * 2
        x = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same'
        )(x)
        x = MaxPooling2D()(x)

    # shape info needed to build decoder model
    shape = x.shape.as_list()

    # generate a 16-dim latent vector
    x = Flatten()(x)
    latent = Dense(latent_dim, name='latent_vector')(x)

    # instantiate encoder model
    encoder = Model(inputs, latent, name='encoder')
    encoder.summary()
    print("\n")
    plot_model(encoder, to_file='classifier-encoder.png', show_shapes=True)

    ##################
    # decoder module #
    ##################

    # build the decoder model
    latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
    x = Dense(shape[1]*shape[2]*shape[3])(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)

    # stack of BN-ReLU-Transposed Conv2D-UpSampling blocks
    for _ in range(2):
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            padding='same'
        )(x)
        x = UpSampling2D()(x)
        filters = int(filters / 2)

    x = Conv2DTranspose(
        filters=1,
        kernel_size=kernel_size,
        padding='same'
    )(x)

    outputs = Activation('sigmoid', name='decoder_output')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    print("\n")
    plot_model(decoder, to_file='classifier-decoder.png', show_shapes=True)

    #####################
    # classifier module #
    #####################

    # classifier model
    latent_inputs = Input(shape=(latent_dim,), name='classifier_input')
    x = Dense(512)(latent_inputs)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(256)(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(num_labels)(x)
    classifier_outputs = Activation('softmax', name='classifier_output')(x)
    classifier = Model(latent_inputs, classifier_outputs, name='classifier')
    classifier.summary()
    print("\n")
    plot_model(classifier, to_file='classifier.png', show_shapes=True)

    # autoencoder = encoder + classifier/decoder
    # instantiate autoencoder model
    autoencoder = Model(inputs,
                        [classifier(encoder(inputs)), decoder(encoder(inputs))],
                        name='autodecoder')
    autoencoder.summary()
    print("\n")

    # plot autoencoder graph
    plot_model(autoencoder, to_file="autoEncode+classifier Model.png", show_shapes=True)

    return autoencoder, encoder, decoder

########
# main #
########

if __name__ == "__main__":

    # MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    num_labels = np.amax(y_train) + 1
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    # get the image size
    input_shape = (image_size, image_size, 1)

    # create the autoencoder + classifier model
    model, encoder, _ = AEclassification(input_shape, filters)

    # compile the model with categorical crossentropy + Mean Square Error (MSE) loss function
    # and Adam as the optimizer
    model.compile(
        loss=[
            'categorical_crossentropy',
            'mse'
        ],
        optimizer='adam',
        metrics=['accuracy', 'mse']
    )

    # train the autoencoder
    model.fit(
        x_train,
        [y_train, x_train],
        validation_data=(x_test, [y_test, x_test]),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

    # predict the autoencoder output from test data
    y_predicted, x_decoded = model.predict(x_test)
    print(np.argmax(y_predicted[:8], axis=1))

    # display the 1st 8 input and decoded images
    imgs = np.concatenate([x_test[:8], x_decoded[:8]])
    imgs = imgs.reshape((4, 4, image_size, image_size))
    imgs = np.vstack([np.hstack(i) for i in imgs])
    plt.figure()
    plt.axis('off')
    plt.title('Input: 1st 2 rows, Decoded: last 2 rows')
    plt.imshow(imgs, interpolation='none', cmap='gray')
    plt.savefig('input_and_decoded.png')
    plt.show()

    latent = encoder.predict(x_test)
    print("Variance:", K.var(K.constant(latent)))
