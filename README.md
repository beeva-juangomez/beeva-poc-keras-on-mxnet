# Keras on MXNet

### Table of contents
1. [Topic](#topic)
2. [Goal](#goal)
3. [Scenarios](#scenarios)
4. [What is Keras?](#what_is_keras)
5. [What is MXNet?](#what_is_mxnet)
6. [Project structure](#project_structure)
7. [Hardware](#hardware)
8. [Result](#result)

### Topic <a name="topic"></a>

To use MXNet as a backend with Keras

### Goal <a name="goal"></a>

To generate results and conclusions that describe the use of MXNet as a backend with Keras in the following cases:

* CPU
* GPU
* Multi-GPU

### Scenarios <a name="scenarios"></a>

* Convolutional NN with LeNet and MNIST
* Recurrent NN with IMDB reviews

### What is Keras? <a name="what_is_keras"></a>

>Keras is a high-level neural networks API, written in Python and capable of
running on top of TensorFlow, CNTK, or Theano. It was developed with a focus on
enabling fast experimentation. Being able to go from idea to result with the
least possible delay is key to doing good research.

The official description of Keras does not include MXNet. Today is not possible
to use MXNet with Keras officially. However, you can find a [fork](https://github.com/dmlc/keras)
on GitHub of Keras 1.2.2 supporting it.

### What is MXNet? <a name="what_is_mxnet"></a>

>MXNet is an modern open-source deep learning framework used to train,
and deploy deep neural networks. It is scalable, allowing for fast model training,
and supports a flexible programming model and multiple languages.

### Project structure <a name="project_structure"></a>

You can find two projects:

* lenet-mnist: Contains an implementation of LeNet architecture (CNN) with MNIST.
* rnn-imdb: Contains an implementation of a RNN with IMDB reviews.

Moreover, in the project root exists a folder with three bash scripts containing the software needed to run both projects, depending on which hardware you have.

Inside both lenet-mnist and rnn-imdb projects you can find scripts that run the neural networks n times with different batch sizes and epochs.

### Hardware <a name="hardware"></a>

AWS EC2 instances:

* c4.xlarge (CPU)
* p2.xlarge (GPU)
* p2.8xlarge (Multi-GPU)


### Result <a name="result"></a>
