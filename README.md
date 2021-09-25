# Advanced Deep Learning with TensorFlow 2 and Keras (2nd Edition)

## Introduction

<p align = "center">
  <img src = "https://raw.githubusercontent.com/hafiz-kamilin/excercise_advancedDeepLearningWithTensorFlow2AndKeras/main/source/book.png" width = "641" height = "790"/>
</p>

Compiled source code from the second edition of [Advanced Deep Learning with TensorFlow 2 and Keras](https://www.packtpub.com/product/advanced-deep-learning-with-tensorflow-2-and-keras-second-edition/9781838821654) book.

IMO, this is a really good introductory book for a widely used machine learning models. But you might need a supplementary free resources available on YouTube, Medium, or research paper to understand more on the content.

## Setup

1. Install Anaconda or Miniconda and create a new environment.
* `conda create --name <your new environment name>`
2. Activate the new environment.
* `conda activate <your new environment name>`
3. Install the required packages.
* `conda install tensorflow-gpu python-pydot graphviz pydot`
4. cd to the [01_testTensorFlowInstallation.py](source/ch01/01_testTensorFlowInstallation.py) directory and run sanity program to test if the installation is properly configured or not.
## Disclaimer

The original source code repository can be located [here](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras). But I have modified some of the code structure to suit my own convenience (i.e. code styling, eliminating spaghetti code, comments, etc).

## [Chapter 1 - Introduction](source/ch01)
1. [Test TensorFlow installation](source/ch01/01_testTensorFlowInstallation.py)
2. [MNIST sampler](source/ch01/02_mnistSampler.py)
3. [MLP on MNIST](source/ch01/03_mnistMLP.py)
4. [CNN on MNIST](source/ch01/04._mnistCNN.py)
5. [RNN on MNIST](source/ch01/05_mnistRNN.py)

## [Chapter 2 - Deep Networks](source/ch02)
6. [Functional API on MNIST](source/ch02/06_functionalCNN.py)
7. [Y-Network on MNIST](source/ch02/07_yNetworkCNN.py)
8. [Shallow Y-Network on MNIST](source/ch02/08_ModifiedYNetworkCNN.py)
9. [ResNet v1 and v2 on CIFAR10](source/ch02/09_resNETcifar10.py)
10. [DenseNet on CIFAR10](source/ch02/10_denseNETcifar10.py)