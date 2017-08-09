# Keras on MXNet

### Motivation

The goal of this PoC is to check the performance of MXNet as a backend with Keras.

* What is Keras?
* What is MXNet?

#### What is Keras?

>Keras is a high-level neural networks API, written in Python and capable of
running on top of TensorFlow, CNTK, or Theano. It was developed with a focus on
enabling fast experimentation. Being able to go from idea to result with the 
least possible delay is key to doing good research

The official description of Keras does not include MXNet. Today is not possible 
to use MXNet with Keras officially. However, you can find a [fork](https://github.com/dmlc/keras)
on GitHub of MXNet 1.2.2 supporting it.

#### What is MXNet?

>MXNet is an modern open-source deep learning framework used to train,
and deploy deep neural networks. It is scalable, allowing for fast model training,
and supports a flexible programming model and multiple languages