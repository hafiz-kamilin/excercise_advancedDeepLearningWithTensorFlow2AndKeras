# Advanced Deep Learning with TensorFlow 2 and Keras (2nd Edition)

## Introduction

<p align = "center">
  <img src = "https://raw.githubusercontent.com/hafiz-kamilin/excercise_advancedDeepLearningWithTensorFlow2AndKeras/main/source/book.png" width = "641" height = "790"/>
</p>

Compiled source code from the second edition of [Advanced Deep Learning with TensorFlow 2 and Keras](https://www.packtpub.com/product/advanced-deep-learning-with-tensorflow-2-and-keras-second-edition/9781838821654) book.

IMO, this is a really good introductory book that give simple and superficial explanation for a widely used machine learning models (feels kinda like [For Dummies series](https://www.dummies.com/) reference books, but skipping most of the steps). 

You will need a supplementary free resources available on YouTube, Medium, or from the research paper itself to understand more on the content.

## Setup

1. Install Anaconda or Miniconda and create a new environment `conda create --name <your new environment name>`.
2. Activate the new environment `conda activate <your new environment name>`.
3. Install the required packages `conda install tensorflow-gpu python-pydot graphviz`.
4. cd to the [01_testTensorFlowInstallation.py](source/ch01/01_testTensorFlowInstallation.py) directory and run sanity program to test if the installation is properly configured or not.

## Disclaimer

The original source code repository can be located [here](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras). But I have modified some of the code structure to suit my own convenience (i.e. code styling, eliminating spaghetti code, comments, optimizing for my low end hardware, etc).

## Chapters

### [1 - Introduction](source/ch01)
1. [Test TensorFlow installation](source/ch01/01_testTensorFlowInstallation.py)
2. [MNIST sampler](source/ch01/02_mnistSampler.py)
3. [MLP on MNIST](source/ch01/03_mnistMLP.py)
4. [CNN on MNIST](source/ch01/04._mnistCNN.py)
5. [RNN on MNIST](source/ch01/05_mnistRNN.py)

### [2 - Deep Networks](source/ch02)
6. [Functional API on MNIST](source/ch02/06_functionalCNN.py)
7. [Y-Network on MNIST](source/ch02/07_yNetworkCNN.py)
8. [Shallow Y-Network on MNIST](source/ch02/08_ModifiedYNetworkCNN.py)
9. [ResNet v1 and v2 on CIFAR10](source/ch02/09_resNETcifar10.py)
10. [DenseNet on CIFAR10](source/ch02/10_denseNETcifar10.py)

### [3 - Autoencoders](source/ch03)
11. [Autoencoder on MNIST](source/ch03/11_MNISTautoencoder.py)
12. [Autoencoders 2-dim latent vector data](source/ch03/12_MNISTautoencoderAE2dim.py)
13. [Autoencoder and classification on MNIST](source/ch03/13_MNISTautoencoderClassification.py)
14. [Autoencoders for denoising image](source/ch03/14_MNISTdenoisingAutoencoder.py)
15. [Autoencoder colorization on monochrome CIFAR10](source/ch03/15_CIFAR10autoencoderColorization.py)

### [4 - Generative Adversarial Network](source/ch04)
16. [DCGAN on MNIST](source/ch04/16_MNISTdcgan.py)
17. [CGAN on MNIST](source/ch04/17_MNISTcgan.py)

### [5 - Improved GAN](source/ch05)
18. [WGAN on MNIST](source/ch05/18_MNISTwgan.py)