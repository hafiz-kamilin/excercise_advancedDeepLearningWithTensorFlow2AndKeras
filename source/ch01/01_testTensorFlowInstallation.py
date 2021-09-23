#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Perform 2 sanity check

1. Check if TensorFlow is usable.
2. Check if GPU can be utilized.

"""

while True:
    option = input("\nDo you want to disable the TensorFlow GPU debug info or not [Yes/No]: ")
    if option == "Yes" or option == "yes" or option == "y":
        # to supress tensorflow-gpu debug information
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        break
    elif option == "No" or option == "no" or option == "n":
        break

# import the required libraries
import tensorflow as tf

# sanity test
print("\nTensorFlow Version: " + str(tf.__version__))
print("Epsilon value: " + str(tf.keras.backend.epsilon()))

# check if the gpu is enabled or not
if tf.test.gpu_device_name(): 

    print("\nGPU is accessible by the TensorFlow: {}.\n".format(tf.test.gpu_device_name()))

else:

   print("\nGPU is not accessible / No compatible GPU found.\n")